# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：评估单个处于孤立状态下的轻量级专家（Lightweight Expert）预测性能。
用于在 MoE 集成前，验证单独利用其中某 1 个降智的小模型来单独生成所有轨迹时的基础能力下界与表现。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `trian_single_expert.py`（模块拼写保持原生）获取轻量级网络的构建函数。
- 依赖全局 `alpamayo_r1` 包。
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

from alpamayo_r1 import helper
from trian_single_expert import AlpamayoTrainDataset, collate_fn, build_model_with_light_expert

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def evaluate(args):
    device = args.device
    dtype = torch.bfloat16

    model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.expert.load_state_dict(ckpt["expert"])
    model.action_in_proj.load_state_dict(ckpt["action_in_proj"])
    model.action_out_proj.load_state_dict(ckpt["action_out_proj"])
    model.eval()
    logger.info("Loaded lightweight expert from %s", args.checkpoint)
    processor = helper.get_processor(model.tokenizer)

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
        min_ade = np.linalg.norm(pred_xy - gt_xyz[None], axis=-1).mean(axis=-1).min()
        coc = extra.get("cot", [[[""]]])[0][0][0] if "cot" in extra else ""

        results.append({"chunk": int(chunk), "clip_id": clip_id, "t0_us": t0_us, "min_ade": min_ade, "coc": coc})
        logger.info("sample %d/%d | clip=%s | minADE %.4f", i + 1, len(dataset), clip_id, min_ade)
        print(f"CoC: {coc}\n")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info("Results saved to %s | mean minADE = %.4f m", args.output, df["min_ade"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--checkpoint", type=str, default="./light_expert_checkpoints/checkpoint-8000/light_expert.pt")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=101)
    parser.add_argument("--chunk-end", type=int, default=110)
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_light_expert_results.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
