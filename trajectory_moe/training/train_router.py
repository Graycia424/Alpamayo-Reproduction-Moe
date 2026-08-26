# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构（非 1.5），训练专注于 Macro-MoE 的多模态语义路由器（Expert Router）。
通过截取 VLM 最后有效特征处的池化隐层张量（pre-CoC hidden state），输入一个 MLP 并利用有监督的聚类标签训练分配能力（路由准确性）。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo_r1` 以及 `train_decoder_expert.py` 内部的数据与初始化接口。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_decoder_expert import (
    AlpamayoTrainDataset,
    build_model_with_light_expert,
    collate_fn,
)
from alpamayo_r1 import helper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class ExpertRouter(nn.Module):
    """MLP router: VLM hidden state -> expert selection logits."""

    def __init__(self, vlm_hidden_dim: int, num_experts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_experts),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class RouterTrainDataset(AlpamayoTrainDataset):
    """Dataset that also returns cluster label for each sample."""

    def __init__(self, cluster_labels_path: str, **kwargs):
        super().__init__(**kwargs)
        labels_df = pd.read_csv(cluster_labels_path)
        self._label_map = {
            (row["clip_id"], int(row["t0_us"])): int(row["cluster_label"])
            for _, row in labels_df.iterrows()
        }
        # Filter to only samples with labels
        self.samples = [(c, t) for c, t in self.samples if (c, t) in self._label_map]
        logger.info("Router dataset: %d samples with cluster labels", len(self.samples))

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        clip_id, t0_us = self.samples[idx]
        item["cluster_label"] = self._label_map[(clip_id, t0_us)]
        return item


def router_collate_fn(batch):
    base = collate_fn(batch)
    base["cluster_label"] = torch.tensor([b["cluster_label"] for b in batch], dtype=torch.long)
    return base


def get_vlm_last_hidden(model, fused_input_ids, attention_mask, vlm_kwargs, device):
    """Extract last valid token's hidden state from VLM prefill."""
    with torch.no_grad():
        vlm_out = model.vlm(
            input_ids=fused_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **vlm_kwargs,
        )
    hidden = vlm_out.hidden_states[-1]  # (B, seq_len, hidden_dim)
    last_pos = attention_mask.sum(dim=1) - 1  # (B,)
    return hidden[torch.arange(hidden.size(0), device=device), last_pos]  # (B, hidden_dim)


def train(args):
    device = args.device
    dtype = torch.bfloat16

    model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    vlm_hidden_dim = model.vlm.config.text_config.hidden_size
    router = ExpertRouter(vlm_hidden_dim, args.num_experts).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    processor = helper.get_processor(model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()

    dataset = RouterTrainDataset(cluster_labels_path=args.cluster_labels, clip_ids=clip_ids)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=router_collate_fn,
    )

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break
            step += 1

            B = batch["ego_history_xyz"].shape[0]
            hist_xyz = batch["ego_history_xyz"].to(device)
            hist_rot = batch["ego_history_rot"].to(device)
            labels = batch["cluster_label"].to(device)

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
                router_input = get_vlm_last_hidden(model, fused_input_ids, attention_mask, vlm_kwargs, device)
                logits = router(router_input)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                logger.info("step %d | ce_loss %.4f | acc %.3f", step, loss.item(), acc)
            if step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"checkpoint-{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(router.state_dict(), save_path / "router.pt")

    final_path = Path(args.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), final_path / "router.pt")
    logger.info("Router training complete. Saved to %s", final_path)


def main():
    parser = argparse.ArgumentParser(description="Train expert router")
    parser.add_argument("--cluster-labels", type=str, default="gt_clustering_results_3/cluster_labels.csv")
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./router_checkpoints")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=179)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
