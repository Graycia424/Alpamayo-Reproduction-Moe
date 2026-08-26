# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构（完全独立于 1.5 体系），针对特定的运动聚类特征（Cluster）训练专属的轨迹解码专家（Decoder Expert）。
通过读取由 `cluster_gt_trajectories.py` 生成的聚类标签过滤数据，完全复用 `train_decoder_expert.py` 的架构和 Flow-Matching (流匹配) 逻辑进行定制化专家训练。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `train_decoder_expert.py` 借用数据集构建和损失计算。
- 依赖全局的 `alpamayo_r1` 包。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_decoder_expert import (
    AlpamayoTrainDataset,
    build_model_with_light_expert,
    collate_fn,
    _compute_fm_loss,
    _save_decoder_expert,
)
from alpamayo_r1 import helper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class ClusterFilteredDataset(AlpamayoTrainDataset):
    """AlpamayoTrainDataset filtered to only include samples from a specific cluster."""

    def __init__(self, cluster_id: int, cluster_labels_path: str, **kwargs):
        super().__init__(**kwargs)
        labels_df = pd.read_csv(cluster_labels_path)
        valid = set(
            zip(labels_df.loc[labels_df["cluster_label"] == cluster_id, "clip_id"],
                labels_df.loc[labels_df["cluster_label"] == cluster_id, "t0_us"])
        )
        self.samples = [(c, t) for c, t in self.samples if (c, t) in valid]
        logger.info("Cluster %d: %d samples after filtering", cluster_id, len(self.samples))


def train(args):
    device = args.device
    dtype = torch.bfloat16

    model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    processor = helper.get_processor(model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()

    dataset = ClusterFilteredDataset(
        cluster_id=args.cluster_id,
        cluster_labels_path=args.cluster_labels,
        clip_ids=clip_ids,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    logger.info("Training expert for cluster %d with %d samples", args.cluster_id, len(dataset))

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break
            step += 1

            B = batch["ego_history_xyz"].shape[0]
            hist_xyz = batch["ego_history_xyz"].to(device)
            hist_rot = batch["ego_history_rot"].to(device)
            fut_xyz = batch["ego_future_xyz"].to(device)
            fut_rot = batch["ego_future_rot"].to(device)

            gt_actions = model.action_space.traj_to_action(
                traj_history_xyz=hist_xyz[:, 0], traj_history_rot=hist_rot[:, 0],
                traj_future_xyz=fut_xyz[:, 0], traj_future_rot=fut_rot[:, 0],
            )

            all_input_ids, all_pixel_values, all_image_grid_thw = [], [], []
            for i in range(B):
                messages = helper.create_message(batch["image_frames"][i])
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, return_dict=True, return_tensors="pt",
                )
                all_input_ids.append(inputs["input_ids"].squeeze(0))
                if "pixel_values" in inputs:
                    all_pixel_values.append(inputs["pixel_values"].squeeze(0))
                if "image_grid_thw" in inputs:
                    all_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))

            max_len = max(ids.shape[0] for ids in all_input_ids)
            pad_id = model.tokenizer.pad_token_id or 0
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros(B, max_len, dtype=torch.long)
            for i, ids in enumerate(all_input_ids):
                L = ids.shape[0]
                input_ids[i, :L] = ids
                attention_mask[i, :L] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            vlm_kwargs: dict[str, Any] = {}
            if all_pixel_values:
                vlm_kwargs["pixel_values"] = torch.cat(all_pixel_values, dim=0).to(device)
            if all_image_grid_thw:
                vlm_kwargs["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0).to(device)

            traj_data = {"ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}

            with torch.autocast(device_type=device, dtype=dtype):
                fused_input_ids = model.fuse_traj_tokens(input_ids, traj_data)
                with torch.no_grad():
                    vlm_out = model.vlm(
                        input_ids=fused_input_ids, attention_mask=attention_mask,
                        use_cache=True, output_hidden_states=False, **vlm_kwargs,
                    )
                prefill_cache = vlm_out.past_key_values
                n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
                loss = _compute_fm_loss(model, gt_actions, prefill_cache, n_diffusion_tokens, device, dtype)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            del prefill_cache

            if step % args.log_every == 0:
                logger.info("step %d | cluster %d | fm_loss %.4f", step, args.cluster_id, loss.item())
            if step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"expert_{args.cluster_id}" / f"checkpoint-{step}"
                _save_decoder_expert(model, save_path)

    final_path = Path(args.output_dir) / f"expert_{args.cluster_id}" / "final"
    _save_decoder_expert(model, final_path)
    logger.info("Cluster %d expert training complete. Saved to %s", args.cluster_id, final_path)


def main():
    parser = argparse.ArgumentParser(description="Train decoder expert for a specific cluster")
    parser.add_argument("--cluster-id", type=int, required=True, help="Cluster ID to train for (0, 1, or 2)")
    parser.add_argument("--cluster-labels", type=str, default="gt_clustering_results_3/cluster_labels.csv")
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./cluster_expert_checkpoints")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=179)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
