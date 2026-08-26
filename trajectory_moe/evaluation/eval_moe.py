# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：全面评估完整的 Mixture-of-Experts (MoE) 架构表现。
该脚本会同步加载 Router 的检查点和全部轻量级 Expert 的权重，计算在海量测试集下 Router 的路由分配命中准确率（基于聚类标签），并评估该聚类切分下的整体 minADE。作为验证 MoE 架构效果的核心评测入口。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `train_router.py`：使用了 `ExpertRouter` 以及 Router 训练时的自定义数据集。
- 依赖 `train_decoder_expert.py`：用来载入轻量级专家的配置设定与初始化入口。
"""

from __future__ import annotations
import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, DynamicCache

from train_decoder_expert import (
    AlpamayoTrainDataset,
    build_model_with_light_expert,
    collate_fn,
    LIGHT_EXPERT_CFG,
    LIGHT_ACTION_IN_PROJ_CFG,
)
from train_router import ExpertRouter, RouterTrainDataset, router_collate_fn, get_vlm_last_hidden
from alpamayo_r1 import helper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _expand_cache(prefill_cache, n: int):
    expanded = DynamicCache()
    for layer_idx, (k, v) in enumerate(prefill_cache):
        expanded.update(k.repeat_interleave(n, dim=0), v.repeat_interleave(n, dim=0), layer_idx)
    return expanded


def sample_from_expert(model, expert, in_proj, out_proj, prefill_cache,
                       hist_xyz, hist_rot, num_samples, device, dtype):
    """Sample trajectories using a specific expert."""
    B = hist_xyz.shape[0]
    n_diff = model.action_space.get_action_space_dims()[0]
    seq_len = prefill_cache.get_seq_length()

    expanded_cache = _expand_cache(prefill_cache, num_samples)
    position_ids = torch.arange(n_diff, device=device)
    position_ids = position_ids[None, None, :].expand(3, B * num_samples, -1).clone() + seq_len
    attn_mask = torch.zeros(
        (B * num_samples, 1, n_diff, seq_len + n_diff), dtype=dtype, device=device
    )

    fwd_kwargs = {}
    if model.config.expert_non_causal_attention:
        fwd_kwargs["is_causal"] = False

    def step_fn(x, t):
        t_bc = t[:, None, None] if t.dim() == 1 else t
        embeds = in_proj(x, t_bc)
        if embeds.dim() == 2:
            embeds = embeds.view(x.shape[0], n_diff, -1)
        out = expert(
            inputs_embeds=embeds, position_ids=position_ids,
            past_key_values=expanded_cache, attention_mask=attn_mask,
            use_cache=True, **fwd_kwargs,
        )
        expanded_cache.crop(seq_len)
        h = out.last_hidden_state[:, -n_diff:]
        return out_proj(h).view(-1, *model.action_space.get_action_space_dims())

    sampled = model.diffusion.sample(batch_size=B * num_samples, step_fn=step_fn, device=device)
    hist_xyz_r = einops.repeat(hist_xyz, "b ... -> (b n) ...", n=num_samples)
    hist_rot_r = einops.repeat(hist_rot, "b ... -> (b n) ...", n=num_samples)
    pred_xyz, _ = model.action_space.action_to_traj(sampled, hist_xyz_r, hist_rot_r)
    return einops.rearrange(pred_xyz, "(b n) ... -> b n ...", n=num_samples)


def evaluate(args):
    device = args.device
    dtype = torch.bfloat16
    import hydra.utils as hyu

    base_model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    # Load experts
    expert_text_config = copy.deepcopy(base_model.vlm.config.text_config)
    for k, v in LIGHT_EXPERT_CFG.items():
        if k != "dtype":
            setattr(expert_text_config, k, v)

    experts, in_projs, out_projs = [], [], []
    for i in range(args.num_experts):
        expert = AutoModel.from_config(expert_text_config).to(device=device, dtype=dtype)
        del expert.embed_tokens
        in_proj = hyu.instantiate(
            LIGHT_ACTION_IN_PROJ_CFG,
            in_dims=base_model.action_space.get_action_space_dims(),
            out_dim=LIGHT_EXPERT_CFG["hidden_size"],
        ).to(device=device, dtype=dtype)
        out_proj = nn.Linear(
            LIGHT_EXPERT_CFG["hidden_size"],
            base_model.action_space.get_action_space_dims()[-1],
        ).to(device=device, dtype=dtype)

        ckpt = torch.load(f"{args.moe_dir}/expert_{i}.pt", map_location=device)
        expert.load_state_dict(ckpt["expert"])
        in_proj.load_state_dict(ckpt["action_in_proj"])
        out_proj.load_state_dict(ckpt["action_out_proj"])
        expert.eval()
        in_proj.eval()
        out_proj.eval()
        experts.append(expert)
        in_projs.append(in_proj)
        out_projs.append(out_proj)

    vlm_hidden_dim = base_model.vlm.config.text_config.hidden_size
    router = ExpertRouter(vlm_hidden_dim, args.num_experts).to(device=device, dtype=dtype)
    router.load_state_dict(torch.load(f"{args.moe_dir}/router.pt", map_location=device))
    router.eval()
    logger.info("Loaded MoE from %s", args.moe_dir)

    processor = helper.get_processor(base_model.tokenizer)
    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    eval_clips = clip_index[chunk_mask]
    clip_ids = eval_clips.index.tolist()

    _labels_df = pd.read_csv(args.cluster_labels)
    _t0_offsets = sorted(_labels_df["t0_us"].unique().tolist())
    dataset = RouterTrainDataset(
        cluster_labels_path=args.cluster_labels, clip_ids=clip_ids,
        t0_offsets_us=_t0_offsets,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=router_collate_fn)

    results = []
    for idx, batch in enumerate(dataloader):
        clip_id, t0_us = dataset.samples[idx]
        gt_label = batch["cluster_label"].item()
        hist_xyz = batch["ego_history_xyz"].to(device)
        hist_rot = batch["ego_history_rot"].to(device)
        gt_xyz = batch["ego_future_xyz"][0, 0, :, :2].numpy()

        messages = helper.create_message(batch["image_frames"][0])
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

        vlm_kwargs: dict[str, Any] = {}
        if "pixel_values" in inputs:
            vlm_kwargs["pixel_values"] = inputs["pixel_values"].to(device)
        if "image_grid_thw" in inputs:
            vlm_kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(device)

        traj_data = {"ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}

        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            fused_ids = base_model.fuse_traj_tokens(input_ids, traj_data)
            vlm_out = base_model.vlm(
                input_ids=fused_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=True, **vlm_kwargs,
            )
            hidden = vlm_out.hidden_states[-1]
            last_pos = attention_mask.sum(dim=1) - 1
            router_input = hidden[torch.arange(1, device=device), last_pos]

            pred_expert = router(router_input).argmax(dim=-1).item()
            prefill_cache = vlm_out.past_key_values

            pred_xyz = sample_from_expert(
                base_model, experts[pred_expert], in_projs[pred_expert], out_projs[pred_expert],
                prefill_cache, hist_xyz[:, 0], hist_rot[:, 0],
                args.num_traj_samples, device, dtype,
            )

        pred_xy = pred_xyz[0, :, :, :2].cpu().numpy()
        min_ade = np.linalg.norm(pred_xy - gt_xyz[None], axis=-1).mean(axis=-1).min()
        correct = int(pred_expert == gt_label)

        results.append({
            "clip_id": clip_id, "t0_us": t0_us,
            "gt_cluster": gt_label, "pred_cluster": pred_expert,
            "router_correct": correct, "min_ade": min_ade,
        })
        logger.info("sample %d/%d | clip=%s | expert=%d(gt=%d) | minADE %.4f",
                     idx + 1, len(dataset), clip_id, pred_expert, gt_label, min_ade)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info("=== Results ===")
    logger.info("Router accuracy: %.3f", df["router_correct"].mean())
    logger.info("Mean minADE: %.4f m", df["min_ade"].mean())
    for c in sorted(df["gt_cluster"].unique()):
        sub = df[df["gt_cluster"] == c]
        logger.info("  Cluster %d: minADE %.4f m (%d samples)", c, sub["min_ade"].mean(), len(sub))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model")
    parser.add_argument("--moe-dir", type=str, default="./moe_checkpoints/checkpoint-1250")
    parser.add_argument("--cluster-labels", type=str, default="gt_clustering_results_199/cluster_labels.csv")
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=180)
    parser.add_argument("--chunk-end", type=int, default=189)
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_moe_results.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
