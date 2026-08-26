# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：针对原版 Alpamayo-R1 基线大模型进行端到端样本的批量推理评定。
负责在庞大的验证 clip-ids 列表里循环预测轨迹，生成文本思维链 (CoC / Chain-of-Thought)，并输出最终的 minADE 指标供对标。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo_r1.models.alpamayo_r1`：即纯原版的 R1 大一统解码网络。
- 依赖 `train_moe.py`：借用其数据装载逻辑构建 Batch。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from train_moe import AlpamayoTrainDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def evaluate(args):
    device = args.device
    dtype = torch.bfloat16

    # Load model using inference.py method
    logger.info("Loading model from %s", args.model_dir)
    model = AlpamayoR1.from_pretrained(str(args.model_dir), dtype=dtype).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Load eval clips from specified chunk range
    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    eval_clips = clip_index[chunk_mask]
    clip_ids = eval_clips.index.tolist()
    logger.info("Evaluating on %d clips from chunks %d-%d", len(clip_ids), args.chunk_start, args.chunk_end)

    dataset = AlpamayoTrainDataset(clip_ids=clip_ids, t0_offsets_us=[2_000_000, 6_000_000, 10_000_000])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    results = []
    for i, batch in enumerate(dataloader):
        clip_id, t0_us = dataset.samples[i]
        chunk = eval_clips.loc[clip_id, "chunk"]

        # Prepare VLM inputs
        messages = helper.create_message(batch["image_frames"][0])
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": batch["ego_history_xyz"],
            "ego_history_rot": batch["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device)

        gt_xyz = batch["ego_future_xyz"][0, 0, :, :2].numpy()

        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                num_traj_samples=args.num_traj_samples,
                return_extra=True,
            )

        pred_xy = pred_xyz[0, 0, :, :, :2].cpu().numpy()
        ade_per_sample = np.linalg.norm(pred_xy - gt_xyz[None], axis=-1).mean(axis=-1)
        min_ade = ade_per_sample.min()

        coc = extra.get("cot", [[[""]]])[0][0] if "cot" in extra else ""

        results.append({
            "chunk": int(chunk),
            "clip_id": clip_id,
            "t0_us": t0_us,
            "min_ade": min_ade,
            "coc": coc,
        })

        logger.info("sample %d/%d | clip=%s | minADE %.4f", i + 1, len(dataset), clip_id, min_ade)
        print(f"CoC: {coc[0]}\n")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info("Results saved to %s", args.output)
    logger.info("=== Evaluation complete: %d samples, mean minADE = %.4f m ===",
                len(results), df["min_ade"].mean())


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alpamayo-R1 with inference.py loading")
    parser.add_argument("--model-dir", type=Path, default=Path("/path/to/models/Alpamayo-R1-10B"))
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=101)
    parser.add_argument("--chunk-end", type=int, default=110)
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_samples_results.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
