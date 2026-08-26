# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo-R1 项目。
主要功能：对 Alpamayo-R1 模型进行端到端的推理论证与指标评估系统。
数据流与模块关系说明：
1. 【输入来源】：利用 `dataset.py` (PhysicalAIAVDatasetInterface) 以及 `alpamayo_r1/helper.py` 提取包含 ground truth（真实环境轨迹、自车运动数据）的真实信息。结合预训练模型（`models/alpamayo_r1.py`等）生成预测结果。
2. 【核心处理】：计算并对比模型预测输出 (Predictions) 轨迹与从底层 `egomotion.py` 及真实环境数据提取出来的 Ground Truth (真实路径点)的差异，主要是 L1/L2 等误差指标（Error metrics）。
3. 【输出去向】：产生的测评聚合分数、距离误差、动作碰撞率指标等，直接输出在控制台或记录日志中供下游分析与训练微调参考。
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from torch import Tensor

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from dataset import PhysicalAIAVDatasetInterface

# Memory optimization for CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    clip_id: str = "5c55dbea-1b22-40ea-8c4f-22199b41502b"
    t0_us: int = 10_000_000  
    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step: float = 0.1 
    num_frames: int = 4  

    extrinsics_parquet: Path = Path(
        "/data/PhysicalAI-Autonomous-Vehicles/calibration/sensor_extrinsics"
        "/sensor_extrinsics.chunk_0000.parquet"
    )
    intrinsics_parquet: Path = Path(
        "/data/PhysicalAI-Autonomous-Vehicles/calibration/camera_intrinsics"
        "/camera_intrinsics.chunk_0000.parquet"
    )
    clip_index_parquet: Path = Path(
        "/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet"
    )
    model_dir: Path = Path("/path/to/models/Alpamayo-R1-10B")
    video_root: Path = Path("/path/to/data/PhysicalAI-Autonomous-Vehicles/camera")
    output_image: Path = Path("frontwide_with_trajectory.jpg")
    extracted_frame: Path = Path("frame_10s.jpg")

    verbose: bool = True
    device: str = "cuda"  
    dtype: torch.dtype = torch.bfloat16
    camera_name: str = "camera_front_wide_120fov"

def extract_frame_cv2(video_path, timestamp_us,output_path,*,verbose = True,) :
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Unable to retrieve FPS from video; check the file.")

    if verbose:
        logger.info("Video FPS: %.3f", fps)

    target_ms = timestamp_us / 1_000.0  
    if verbose:
        logger.info("Target timestamp: %d ?s (%.3f ms)", timestamp_us, target_ms)

    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ret, frame = cap.read()
    if not ret:
        frame_idx = int(timestamp_us / 1_000_000 * fps)
        if verbose:
            logger.warning(
                "Time?based seek failed, falling back to frame index %d", frame_idx
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp_us} ?s.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image to {output_path}")

    if verbose:
        logger.info("Frame saved to %s", output_path)


def quat_to_rot(x, y, z, w):
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero?norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def world_to_cam(pt_world, R_wc, t_wc):
    return R_wc.T @ (pt_world - t_wc)


def project_fisheye( pt_cam, fx_poly, cx, cy,):
    X, Y, Z = pt_cam
    if Z <= 0:
        return None
    r_xy = np.hypot(X, Y)  
    theta = np.arctan2(r_xy, Z) 
    radius = np.polyval(fx_poly, theta)

    if r_xy == 0:
        u, v = cx, cy
    else:
        u = cx + radius * (X / r_xy)
        v = cy + radius * (Y / r_xy)

    return np.array([u, v], dtype=np.float64)

def load_extrinsics(parquet_path, clip_id, sensor_name) :
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, sensor_name)]
    qx, qy, qz, qw = row[["qx", "qy", "qz", "qw"]].astype(float).values
    tx, ty, tz = row[["x", "y", "z"]].astype(float).values
    logger.info("Extrinsics for %s �C quaternion: %s, translation: %s", sensor_name,
                (qx, qy, qz, qw), (tx, ty, tz))
    return np.array([qx, qy, qz, qw], dtype=np.float64), np.array([tx, ty, tz], dtype=np.float64)

def load_intrinsics(parquet_path, clip_id, camera_name) :
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, camera_name)]
    width, height, cx, cy = row[["width", "height", "cx", "cy"]].astype(float).values
    fw_coeffs = row[
        ["fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4"]
    ].astype(float).values
    fw_poly = fw_coeffs[::-1]
    logger.info(
        "Intrinsics for %s �C size: %dx%d, principal point: (%.2f, %.2f)",
        camera_name,
        int(width),
        int(height),
        cx,
        cy,
    )
    return int(width), int(height), float(cx), float(cy), fw_poly

def load_clip_index(clip_index_parquet, chunk_id=0, num_clips=100):
    """
    Load clip_ids from clip_index.parquet for a specific chunk.
    
    Args:
        clip_index_parquet: Path to clip_index.parquet
        chunk_id: Chunk ID to load (default: 0 for first chunk)
        num_clips: Number of clips to load (default: 100)
    
    Returns:
        List of clip_ids for the specified chunk, limited to num_clips
    """
    df = pd.read_parquet(clip_index_parquet)
    
    # Filter by chunk_id (assuming column name 'chunk' or similar)
    if 'chunk' in df.columns:
        chunk_df = df[df['chunk'] == chunk_id]
    else:
        # If index is multi-level with chunk info, try getting it from index
        chunk_df = df[df.index.get_level_values('chunk') == chunk_id] if 'chunk' in df.index.names else df
    
    clip_ids = chunk_df['clip_id'].tolist() if 'clip_id' in chunk_df.columns else chunk_df.index.get_level_values('clip_id').tolist()
    
    logger.info("Loaded %d clip_ids from chunk %d", len(clip_ids), chunk_id)
    return clip_ids[:num_clips]


def compute_minADE_for_clip(cfg, model, processor, clip_id) -> float:
    """
    Run inference on a single clip and compute minADE.
    
    Args:
        cfg: Config object
        model: The AlpamayoR1 model
        processor: The processor/tokenizer
        clip_id: The clip_id to process
    
    Returns:
        minADE value for this clip, or None if inference fails
    """
    try:
        cfg.clip_id = clip_id
        
        # Build dataset
        data = build_dataset(cfg)
        
        # Run inference with no_grad to save memory
        with torch.no_grad():
            pred_xyz, pred_rot, extra = run_model(
                model=model,
                processor=processor,
                data=data,
                device=cfg.device,
                dtype=cfg.dtype,
            )
        
        # Compute minADE
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        
        logger.info("clip_id: %s, minADE: %.4f m", clip_id, min_ade)
        
        # Clean up memory
        del data, pred_xyz, pred_rot, extra
        torch.cuda.empty_cache()
        
        return min_ade
        
    except Exception as e:
        logger.warning("Failed to process clip_id %s: %s", clip_id, str(e))
        torch.cuda.empty_cache()
        return None

def build_dataset(cfg) :
    avdi = PhysicalAIAVDatasetInterface()
    sensor_name = cfg.camera_name 
    q, t = load_extrinsics(cfg.extrinsics_parquet, cfg.clip_id, sensor_name)
    R_wc = quat_to_rot(*q)  
    t_wc = t

    width, height, cx, cy, fw_poly = load_intrinsics(
        cfg.intrinsics_parquet, cfg.clip_id, sensor_name
    )

    egomotion = avdi.get_clip_feature(
        cfg.clip_id,
        avdi.features.LABELS.EGOMOTION,
        types="egomotion",
    )

    assert (
        cfg.t0_us
        > cfg.num_history_steps * cfg.time_step * 1_000_000
    ), "t0_us must be larger than the history time range"

    history_offsets_us = np.arange(
        -(cfg.num_history_steps - 1) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000 / 2,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = cfg.t0_us + history_offsets_us

    future_offsets_us = np.arange(
        cfg.time_step * 1_000_000,
        (cfg.num_future_steps + 0.5) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = cfg.t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation  
    ego_history_quat = ego_history.pose.rotation.as_quat()  

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation  
    ego_future_quat = ego_future.pose.rotation.as_quat() 

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)
    ).as_matrix()
    ego_future_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)
    ).as_matrix()

    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)
    )

    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    image_timestamps = np.array(
        [
            cfg.t0_us
            - (cfg.num_frames - 1 - i) * int(cfg.time_step * 1_000_000)
            for i in range(cfg.num_frames)
        ],
        dtype=np.int64,
    )

    image_frames_list: List[Tensor] = []
    camera_indices_list: List[int] = []
    timestamps_list: List[Tensor] = []

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(
            cfg.clip_id,
            cam_feature,
            types="camera",
        )
        frames_np, frame_ts = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames_np)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1].lower()
        else:
            raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_ts.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6 

    data = {
        "image_frames": image_frames, 
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor, 
        "ego_history_rot": ego_history_rot_tensor, 
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor, 
        "relative_timestamps": relative_timestamps, 
        "absolute_timestamps": all_timestamps, 
        "t0_us": cfg.t0_us,
        "clip_id": cfg.clip_id,
        "intrinsics": {
            "width": width,
            "height": height,
            "cx": cx,
            "cy": cy,
            "fw_poly": fw_poly,
        },
        "extrinsics": {"R_wc": R_wc, "t_wc": t_wc},
    }

    logger.info("Dataset built successfully.")
    return data

def run_model(model, processor, data, device, dtype,) :
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    torch.cuda.manual_seed_all(42)
    
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=6,
                max_generation_length=256,
                return_extra=True,
            )
    return pred_xyz, pred_rot, extra

def draw_trajectory_on_image(img_path, output_path, pred_xyz, intrinsics, extrinsics, width, height,) :
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    R_wc = extrinsics["R_wc"]
    t_wc = extrinsics["t_wc"]
    fx_poly = intrinsics["fw_poly"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    pred_pts_world = pred_xyz.squeeze().cpu().numpy() 

    pixel_points: List[Tuple[int, int]] = []
    for pt_w in pred_pts_world:
        pt_c = world_to_cam(pt_w, R_wc, t_wc)
        proj = project_fisheye(pt_c, fx_poly, cx, cy)
        if proj is None:
            continue
        u, v = proj
        if 0 <= u < width and 0 <= v < height:
            pixel = (int(round(u)), int(round(v)))
            pixel_points.append(pixel)
            cv2.circle(img, pixel, radius=3, color=(0, 0, 255), thickness=-1) 

    if len(pixel_points) >= 2:
        cv2.polylines(
            img,
            [np.array(pixel_points, dtype=np.int32)],
            isClosed=False,
            color=(0, 255, 0),
            thickness=2,
        )
    cv2.imwrite(str(output_path), img)
    logger.info("Trajectory visualisation saved to %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo?R1 trajectory prediction on single or multiple clips."
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Single clip_id to process. If not provided, processes first clips from chunk"
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=0,
        help="Chunk ID to load clips from (default: 0)"
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=100,
        help="Number of clips to process (default: 100)"
    )
    parser.add_argument(
        "--clip-index",
        type=Path,
        default=Path("/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet"),
        help="Path to clip_index.parquet"
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/path/to/models/Alpamayo-R1-10B"),
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("frontwide_with_trajectory.jpg"),
    )
    parser.add_argument(
        "--extracted-frame",
        type=Path,
        default=Path("frame_10s.jpg"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    cfg = Config(
        t0_us=args.t0_us,
        model_dir=args.model_dir,
        output_image=args.output_image,
        extracted_frame=args.extracted_frame,
        verbose=args.verbose,
        clip_index_parquet=args.clip_index,
    )

    # Load model once (reuse for all clips)
    logger.info("Loading Alpamayo?R1 model from %s", cfg.model_dir)
    
    # Clear GPU memory before loading
    torch.cuda.empty_cache()
    
    model = AlpamayoR1.from_pretrained(
        str(cfg.model_dir), dtype=cfg.dtype
    ).to(cfg.device)
    model.eval()  # Set to evaluation mode
    processor = helper.get_processor(model.tokenizer)
    
    # Determine which clips to process
    if args.clip_id:
        # Single clip mode
        clip_ids = [args.clip_id]
        logger.info("Processing single clip: %s", args.clip_id)
    else:
        # Batch mode: load clip_ids from clip_index.parquet
        try:
            clip_ids = load_clip_index(
                cfg.clip_index_parquet,
                chunk_id=args.chunk_id,
                num_clips=args.num_clips
            )
            logger.info("Processing %d clips from chunk %d", len(clip_ids), args.chunk_id)
        except Exception as e:
            logger.error("Failed to load clip_index: %s", str(e))
            logger.info("Falling back to default clip_id")
            clip_ids = [cfg.clip_id]
    
    # Process clips and collect minADE values
    min_ade_values = []
    successful_clips = 0
    failed_clips = 0
    
    for idx, clip_id in enumerate(clip_ids, 1):
        logger.info("Processing clip %d/%d: %s", idx, len(clip_ids), clip_id)
        
        try:
            min_ade = compute_minADE_for_clip(cfg, model, processor, clip_id)
            if min_ade is not None:
                min_ade_values.append(min_ade)
                successful_clips += 1
            else:
                failed_clips += 1
        except Exception as e:
            logger.warning("Error processing clip %s: %s", clip_id, str(e))
            failed_clips += 1
    
    # Report results
    logger.info("=" * 80)
    logger.info("Processing completed: %d successful, %d failed", successful_clips, failed_clips)
    
    if min_ade_values:
        avg_min_ade = np.mean(min_ade_values)
        std_min_ade = np.std(min_ade_values)
        min_min_ade = np.min(min_ade_values)
        max_min_ade = np.max(min_ade_values)
        
        logger.info("Average minADE: %.4f m", avg_min_ade)
        logger.info("Std minADE: %.4f m", std_min_ade)
        logger.info("Min minADE: %.4f m", min_min_ade)
        logger.info("Max minADE: %.4f m", max_min_ade)
        logger.info("=" * 80)
        
        print("\nFinal Results:")
        print("  Average minADE: {:.4f} m".format(avg_min_ade))
        print("  Std minADE: {:.4f} m".format(std_min_ade))
        print("  Min minADE: {:.4f} m".format(min_min_ade))
        print("  Max minADE: {:.4f} m".format(max_min_ade))
        print("  Processed clips: {}/{}".format(successful_clips, len(clip_ids)))
    else:
        logger.error("No successful clips processed!")
    
    logger.info("All done!")

if __name__ == "__main__":
    main()