# -*- coding: utf-8 -*-
"""
批量顺序运行多个 clip 的推理脚本，模型只加载一次以加快多 clip 推理流程。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import traceback

import pandas as pd
import torch
import time
from datetime import datetime, timedelta

from dataclasses import asdict

from inference_ade import Config, build_dataset, run_model, compute_min_ade_6
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.viz_utils import plot_bev_comparison, make_camera_grid
import numpy as np

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s", stream=sys.stdout)

logging.getLogger("inference_ade").setLevel(logging.WARNING)
logging.getLogger("dataset").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple clips sequentially reusing a single model load.")
    p.add_argument("--clip-ids", type=str, default=None, help="Comma-separated clip ids to process")
    p.add_argument("--clips-file", type=Path, default=None, help="Path to a file containing one clip id per line")
    p.add_argument("--clip-index", type=Path, default=Path("/path/to/data/parquet/clip_index.parquet"), help="Path to clip_index.parquet")
    p.add_argument("--chunk-ids", type=str, default="10,20,30,40,50,60,70,80,90,100", help="Comma-separated chunk ids to load clips from")
    p.add_argument("--num-clips", type=int, default=100, help="Number of clips to load per chunk")
    p.add_argument("--model-dir", type=Path, default=Path("/path/to/data/Alpamayo-1.5-10B"), help="Pretrained model directory")
    p.add_argument("--cameras-config", type=str, default="1_cam", choices=["1_cam", "3_cam", "4_cam", "cross_front_rl", "cross_front_rr", "cross_front_rt", "7_cam"])
    p.add_argument("--visualize", action="store_true", help="Save BEV visualization")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    p.add_argument("--results-csv", type=Path, default=None, help="Optional CSV file")
    p.add_argument("--exp-dir", type=Path, default=None, help="Optional experiment directory")
    p.add_argument("--add-timestamp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default="cuda:1")
    return p.parse_args()


def load_clip_list(args):
    if args.clip_ids:
        return [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    if args.clips_file and args.clips_file.exists():
        with open(args.clips_file, "r", encoding="utf-8") as f:
            return [l.strip() for l in f.readlines() if l.strip()]
    return []


def load_clip_index_local(clip_index_parquet: Path, chunk_id: int, num_clips: int = 100):
    df = pd.read_parquet(clip_index_parquet)
    if 'chunk' in df.columns:
        chunk_df = df[df['chunk'] == chunk_id]
    else:
        chunk_df = df[df.index.get_level_values('chunk') == chunk_id]
        
    if 'clip_id' in chunk_df.columns:
        clip_ids = chunk_df['clip_id'].tolist()
    elif 'clip_id' in chunk_df.index.names:
        clip_ids = chunk_df.index.get_level_values('clip_id').tolist()
    else:
        clip_ids = chunk_df.index.tolist()
        
    return [(cid, chunk_id) for cid in clip_ids[:num_clips]]

def main():
    args = parse_args()
    clip_list = load_clip_list(args)
    
    if len(clip_list) == 0 and args.chunk_ids is not None:
        if args.clip_index is None:
            raise ValueError("--clip-index must be provided")
        chunk_ids = [int(c.strip()) for c in args.chunk_ids.split(",") if c.strip()]
        for cid in chunk_ids:
            cls = load_clip_index_local(args.clip_index, cid, args.num_clips)
            clip_list.extend(cls)
            
    if len(clip_list) == 0:
        raise ValueError("No clips specified.")

    base_cfg = Config()
    base_cfg.model_dir = args.model_dir
    base_cfg.device = args.device

    # Remove the global calibration path setting for first_chunk_id here, since we do it per clip now.
    try:
        logger.info("CLI args (raw): %s", vars(args))
    except Exception:
        pass

    if args.exp_dir is not None:
        base_out = args.exp_dir
    else:
        base_out = args.output_dir

    if args.add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = base_out / ts

    base_out.mkdir(parents=True, exist_ok=True)
    args.output_dir = base_out

    if args.results_csv is None:
        default_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" if args.add_timestamp else "results.csv"
        args.results_csv = args.output_dir / default_name
    else:
        if args.add_timestamp:
            stem = args.results_csv.stem
            suffix = args.results_csv.suffix
            args.results_csv = args.results_csv.with_name(f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}")

    if not args.results_csv.exists():
        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_csv, "w", encoding="utf-8") as f:
            f.write("clip_id,ade_no_nav\n")

    log_path = args.output_dir / "run.log"
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # 2. 文件输出也去掉了 %(name)s
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_path) for h in logger.handlers):
        logger.addHandler(fh)

    cam_mapping = {
        "1_cam": ["camera_front_wide_120fov"],
        "2_cam": ["camera_front_wide_120fov", "camera_front_tele_30fov"],
        "4_cam": ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_front_tele_30fov"],
        "front_rl": ["camera_front_wide_120fov", "camera_rear_left_70fov"],
        "front_rr": ["camera_front_wide_120fov", "camera_rear_right_70fov"],
        "front_rt": ["camera_front_wide_120fov", "camera_rear_tele_30fov"],
        "cross_front_rl": ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_rear_left_70fov"],
        "cross_front_rr": ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_rear_right_70fov"],
        "cross_front_rt": ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_rear_tele_30fov"],
    }
    loaded_cam_names = cam_mapping.get(args.cameras_config, [args.cameras_config])
    logger.info("============= RUN INFO =============")
    logger.info("Cameras config: %s", args.cameras_config)
    logger.info("Specific cameras: %s", ", ".join(loaded_cam_names))
    logger.info("====================================")

    logger.info("Loading model from %s", args.model_dir)
    model = Alpamayo1_5.from_pretrained(str(args.model_dir), dtype=base_cfg.dtype).to(base_cfg.device)
    processor = helper.get_processor(model.tokenizer)
    model.eval()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    results = []
    start_time = time.time()

    for i, item in enumerate(clip_list, start=1):
        if isinstance(item, tuple):
            clip_id, chunk_id = item
            chunk_str = f"{chunk_id:04d}"
            base_cfg.extrinsics_parquet = Path(f"/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/sensor_extrinsics/sensor_extrinsics.chunk_{chunk_str}.parquet")
            base_cfg.intrinsics_parquet = Path(f"/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk_str}.parquet")
        else:
            clip_id = item
            
        ade_val = 0.0
        try:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            cfg = Config(
                clip_id=clip_id,
                t0_us=base_cfg.t0_us,
                num_history_steps=base_cfg.num_history_steps,
                num_future_steps=base_cfg.num_future_steps,
                time_step=base_cfg.time_step,
                num_frames=base_cfg.num_frames,
                extrinsics_parquet=base_cfg.extrinsics_parquet,
                intrinsics_parquet=base_cfg.intrinsics_parquet,
                model_dir=base_cfg.model_dir,
                video_root=base_cfg.video_root,
                output_image=base_cfg.output_image,
                nav_text=base_cfg.nav_text,
                nav_text_swapped=base_cfg.nav_text_swapped,
                verbose=base_cfg.verbose,
                device=base_cfg.device,
                dtype=base_cfg.dtype,
                visualize=args.visualize,
            )
            
            if args.cameras_config is not None:
                cfg.cameras_config = args.cameras_config
                
            cfg.camera_name = "camera_front_wide_120fov"

            if i == 1:
                try:
                    logger.info("First Clip Config: %s", asdict(cfg))
                except Exception:
                    pass

            data = build_dataset(cfg)

            with torch.no_grad():
                pred_no_nav, _, extra = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=None)

            arr = pred_no_nav.cpu().numpy()
            if arr.ndim < 4:
                raise RuntimeError(f"Unexpected pred_no_nav shape: {arr.shape}")

            ade_no_nav = compute_min_ade_6(arr, data["ego_future_xyz"].cpu().numpy())
            
            # 3. 提取干净的数字，不再输出多余的打印
            ade_val = float(ade_no_nav[0]) if ade_no_nav is not None else 0.0

            if args.visualize:
                camera_grid = make_camera_grid(data["image_frames"], data["camera_indices"])
                fig = plot_bev_comparison(
                    pred_with_nav=None,
                    pred_no_nav=pred_no_nav,
                    pred_counterfactual=None,
                    nav_text=None,
                    nav_text_swapped=None,
                    gt_future_xyz=data["ego_future_xyz"],
                    camera_images=camera_grid,
                    title=f"BEV - {clip_id}",
                )
                out_path = args.output_dir / f"{clip_id}_bev.png"
                fig.savefig(str(out_path), dpi=150, bbox_inches="tight")

            results.append((clip_id, ade_val))
            
            if args.results_csv is not None:
                with open(args.results_csv, "a", encoding="utf-8") as f:
                    f.write(f"{clip_id},{ade_val}\n")

        except Exception:
            if args.verbose:
                logger.error("Failed processing clip %s:\n%s", clip_id, traceback.format_exc())
            else:
                logger.error("Failed processing clip %s: %s", clip_id, str(sys.exc_info()[1]))
            results.append((clip_id, None))
            ade_val = -1.0 
        
        # 4. 单行三合一输出
        elapsed_time = time.time() - start_time
        avg_time_per_clip = elapsed_time / i
        remaining_clips = len(clip_list) - i
        eta_seconds = int(avg_time_per_clip * remaining_clips)
        eta_str = str(timedelta(seconds=eta_seconds))
        
        if ade_val >= 0:
            logger.info("[%d/%d] Clip: %s | ADE: %.4f | 耗时: %.1fs | ETA: %s", i, len(clip_list), clip_id, ade_val, elapsed_time, eta_str)
        else:
            logger.info("[%d/%d] Clip: %s | ADE: FAILED | 耗时: %.1fs | ETA: %s", i, len(clip_list), clip_id, elapsed_time, eta_str)

    logger.info("All clips processed. Summary:")
    ade_values = [ade for _, ade in results if ade is not None and ade >= 0]
    successful = len(ade_values)
    failed = len(results) - successful

    if ade_values:
        avg_min_ade = float(np.mean(ade_values))
        logger.info("Aggregate results - Success: %d, Failed: %d", successful, failed)
        logger.info("Final Average minADE: %.4f m | Config: %s | Cameras: %s", avg_min_ade, args.cameras_config, ", ".join(loaded_cam_names))
        print(f"\nFinal Results:\n  Average minADE: {avg_min_ade:.4f} m\n  Config: {args.cameras_config}\n  Cameras: {', '.join(loaded_cam_names)}\n  Processed clips: {successful}/{len(results)}")
    else:
        logger.error("No successful clips processed!")


if __name__ == "__main__":
    main()