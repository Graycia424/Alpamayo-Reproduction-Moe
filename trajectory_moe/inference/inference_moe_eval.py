# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构的 Macro-MoE 批量推理评测脚本。
能够在给定的数据集 Chunk 范围内，遍历选取 Clip，逐个调用拥有路由分发机制的 MoE 模型生成轨迹，并汇总计算最终的 minADE 均值与方差。
注意：此脚本明确基于最新的 R1 模型（AlpamayoR1MoE），而非 1.5 版本的基线。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo_r1.models.alpamayo_r1_moe`（最新研发的包含轻量专家和路由的 R1 MoE 模型）。
- 依赖 `inference` （用于复用一些基础的数据封装配置）。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpamayo_r1.models.alpamayo_r1_moe import AlpamayoR1MoE
from alpamayo_r1 import helper
from inference import Config, build_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def compute_minADE_moe(cfg, model, processor, clip_id, num_traj_samples=6):
    """Run MoE inference on a single clip and compute minADE."""
    try:
        cfg.clip_id = clip_id
        data = build_dataset(cfg)

        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        model_inputs = helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, cfg.device)

        with torch.no_grad(), torch.autocast(device_type=cfg.device, dtype=cfg.dtype):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=num_traj_samples, max_generation_length=256, return_extra=True,
            )

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        min_ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1).min()

        logger.info("clip_id: %s, minADE: %.4f m, expert: %s", clip_id, min_ade, extra.get("expert_idx"))

        del data, pred_xyz, pred_rot, extra, model_inputs
        torch.cuda.empty_cache()
        return min_ade

    except Exception as e:
        logger.warning("Failed clip %s: %s", clip_id, str(e))
        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser(description="Alpamayo-R1 MoE batch inference")
    parser.add_argument("--checkpoint", type=Path, default="./moe_checkpoints/final", help="Path to MoE checkpoint")
    parser.add_argument("--clip-index", type=str, default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=101)
    parser.add_argument("--chunk-end", type=int, default=103)
    parser.add_argument("--num-clips", type=int, default=10, help="Limit number of clips (default: all)")
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--t0-us", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config(t0_us=args.t0_us)
    cfg.device = args.device

    # Load clip_ids from chunk range
    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[mask].index.tolist()
    if args.num_clips:
        clip_ids = clip_ids[:args.num_clips]
    logger.info("Loaded %d clips from chunks %d–%d", len(clip_ids), args.chunk_start, args.chunk_end)

    # Load model
    logger.info("Loading MoE model from %s", args.checkpoint)
    model = AlpamayoR1MoE.from_pretrained(str(args.checkpoint), dtype=cfg.dtype).to(cfg.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Evaluate
    min_ade_values = []
    for idx, clip_id in enumerate(clip_ids, 1):
        logger.info("Processing %d/%d: %s", idx, len(clip_ids), clip_id)
        result = compute_minADE_moe(cfg, model, processor, clip_id, args.num_traj_samples)
        if result is not None:
            min_ade_values.append(result)

    # Report
    logger.info("=" * 80)
    if min_ade_values:
        print("\nFinal Results:")
        print(f"  Average minADE: {np.mean(min_ade_values):.4f} m")
        print(f"  Std minADE: {np.std(min_ade_values):.4f} m")
        print(f"  Min minADE: {np.min(min_ade_values):.4f} m")
        print(f"  Max minADE: {np.max(min_ade_values):.4f} m")
        print(f"  Processed clips: {len(min_ade_values)}/{len(clip_ids)}")
    else:
        logger.error("No successful clips!")
    logger.info("Done.")


if __name__ == "__main__":
    main()


# python inference_moe_evaluation.py --checkpoint ./moe_checkpoints/final \
#    --chunk-start 0 --chunk-end 2 \
#    --num-clips 50 \
#    --num-traj-samples 6
