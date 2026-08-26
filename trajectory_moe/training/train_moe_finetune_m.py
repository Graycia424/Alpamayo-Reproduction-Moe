# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构的 Macro-MoE 第三阶段核心任务：联合微调 (Joint Fine-tuning，多卡 DDP 并行训练版本)。
将之前冻结的 VLM 视为特征提取与思维链层，联合全新初始化的 1个 Router 与 N 个预训练的 Experts，
利用 Straight-Through Gumbel-Softmax 实现端到端的离散专家分配计算（合并交叉熵和流匹配回归 Loss）。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `train_router.py` 以获取路由器定义模型和 Router 的微调数据集。
- 依赖 `train_decoder_expert.py` 从而重用轻量级专家的配置设定与流匹配环境。
"""

from __future__ import annotations
import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel

from train_decoder_expert import (
    AlpamayoTrainDataset,
    build_model_with_light_expert,
    collate_fn,
    LIGHT_EXPERT_CFG,
    LIGHT_ACTION_IN_PROJ_CFG,
)
from train_router import ExpertRouter, RouterTrainDataset, router_collate_fn, get_vlm_last_hidden
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def build_moe_model(model_dir: str, expert_paths: list[str], router_path: str,
                     num_experts: int, device: str, dtype: torch.dtype):
    """Build MoE model: VLM (frozen) + N experts + router (all trainable)."""
    import hydra.utils as hyu

    # Build base model to get VLM and action_space
    base_model = build_model_with_light_expert(model_dir, device, dtype)

    # Build N expert copies and load pretrained weights
    experts, in_projs, out_projs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
    expert_text_config = copy.deepcopy(base_model.vlm.config.text_config)
    for k, v in LIGHT_EXPERT_CFG.items():
        if k != "dtype":
            setattr(expert_text_config, k, v)

    for i, epath in enumerate(expert_paths):
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

        ckpt = torch.load(epath, map_location=device)
        expert.load_state_dict(ckpt["expert"])
        in_proj.load_state_dict(ckpt["action_in_proj"])
        out_proj.load_state_dict(ckpt["action_out_proj"])
        experts.append(expert)
        in_projs.append(in_proj)
        out_projs.append(out_proj)
        logger.info("Loaded expert %d from %s", i, epath)

    # Load router
    vlm_hidden_dim = base_model.vlm.config.text_config.hidden_size
    router = ExpertRouter(vlm_hidden_dim, num_experts).to(device=device, dtype=dtype)
    router.load_state_dict(torch.load(router_path, map_location=device))
    logger.info("Loaded router from %s", router_path)

    # Freeze VLM, unfreeze experts + router
    for p in base_model.parameters():
        p.requires_grad = False
    for m in [experts, in_projs, out_projs, router]:
        for p in m.parameters():
            p.requires_grad = True

    return base_model, experts, in_projs, out_projs, router


def compute_expert_fm_loss(model, expert, in_proj, out_proj, gt_actions, prefill_cache,
                           n_diffusion_tokens, device, dtype):
    """Compute flow-matching loss for a single expert."""
    B = gt_actions.shape[0]
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(gt_actions)
    t_bc = t[:, None, None]
    x_t = (1 - t_bc) * noise + t_bc * gt_actions
    target = gt_actions - noise

    prefill_seq_len = prefill_cache.get_seq_length()
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = position_ids[None, None, :].expand(3, B, -1).clone() + prefill_seq_len

    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=dtype, device=device,
    )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    embeds = in_proj(x_t, t_bc)
    if embeds.dim() == 2:
        embeds = embeds.view(B, n_diffusion_tokens, -1)

    expert_out = expert(
        inputs_embeds=embeds, position_ids=position_ids,
        past_key_values=prefill_cache, attention_mask=attention_mask,
        use_cache=True, **forward_kwargs,
    )
    prefill_cache.crop(prefill_seq_len)

    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred = out_proj(last_hidden).view(B, *model.action_space.get_action_space_dims())
    return F.mse_loss(pred, target)


def train(args):
    ddp = dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        rank, world_size = setup_ddp()
        device = f"cuda:{rank}"
    else:
        rank, world_size = 0, 1
        device = args.device
    is_main = rank == 0
    dtype = torch.bfloat16

    expert_paths = [
        f"{args.expert_dir}/expert_{i}/final/decoder_expert.pt" for i in range(args.num_experts)
    ]
    base_model, experts, in_projs, out_projs, router = build_moe_model(
        args.base_model_dir, expert_paths, args.router_path, args.num_experts, device, dtype
    )

    if ddp:
        router = DDP(router, device_ids=[rank])
        experts = nn.ModuleList([DDP(e, device_ids=[rank]) for e in experts])
        in_projs = nn.ModuleList([DDP(m, device_ids=[rank]) for m in in_projs])
        out_projs = nn.ModuleList([DDP(m, device_ids=[rank]) for m in out_projs])

    all_params = list(router.parameters())
    for m in [experts, in_projs, out_projs]:
        all_params.extend(m.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    processor = helper.get_processor(base_model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()

    dataset = RouterTrainDataset(cluster_labels_path=args.cluster_labels, clip_ids=clip_ids)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=router_collate_fn, sampler=sampler,
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
            fut_xyz = batch["ego_future_xyz"].to(device)
            fut_rot = batch["ego_future_rot"].to(device)
            labels = batch["cluster_label"].to(device)

            gt_actions = base_model.action_space.traj_to_action(
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
            pad_id = base_model.tokenizer.pad_token_id or 0
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

            autocast_device = device.split(":")[0]
            with torch.autocast(device_type=autocast_device, dtype=dtype):
                fused_input_ids = base_model.fuse_traj_tokens(input_ids, traj_data)

                # Get VLM hidden state and KV cache in one pass
                with torch.no_grad():
                    vlm_out = base_model.vlm(
                        input_ids=fused_input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=True, **vlm_kwargs,
                    )
                hidden = vlm_out.hidden_states[-1]
                last_pos = attention_mask.sum(dim=1) - 1
                router_input = hidden[torch.arange(B, device=device), last_pos]
                prefill_cache = vlm_out.past_key_values

                # Router loss
                logits = router(router_input)
                loss_routing = F.cross_entropy(logits, labels)

                # Trajectory loss: Gumbel-Softmax weighted sum of per-expert FM losses
                weights = F.gumbel_softmax(logits, tau=args.gumbel_tau, hard=False)
                n_diff = base_model.action_space.get_action_space_dims()[0]

                loss_traj = torch.tensor(0.0, device=device, dtype=dtype)
                for eid in range(args.num_experts):
                    expert_loss = compute_expert_fm_loss(
                        base_model, experts[eid], in_projs[eid], out_projs[eid],
                        gt_actions, prefill_cache, n_diff, device, dtype,
                    )
                    loss_traj = loss_traj + weights[:, eid].mean() * expert_loss

                loss = loss_routing + args.alpha * loss_traj

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            del prefill_cache

            if is_main and step % args.log_every == 0:
                acc = (logits.argmax(-1) == labels).float().mean().item()
                logger.info("step %d | loss %.4f (route %.4f + traj %.4f) | acc %.3f",
                            step, loss.item(), loss_routing.item(), loss_traj.item(), acc)
            if is_main and step % args.save_every == 0:
                _save_moe(router, experts, in_projs, out_projs,
                          Path(args.output_dir) / f"checkpoint-{step}")

    if is_main:
        _save_moe(router, experts, in_projs, out_projs, Path(args.output_dir) / "final")
        logger.info("MoE fine-tuning complete.")
    if ddp:
        dist.destroy_process_group()


def _unwrap(m):
    return m.module if isinstance(m, DDP) else m


def _save_moe(router, experts, in_projs, out_projs, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(_unwrap(router).state_dict(), path / "router.pt")
    for i in range(len(experts)):
        torch.save({
            "expert": _unwrap(experts[i]).state_dict(),
            "action_in_proj": _unwrap(in_projs[i]).state_dict(),
            "action_out_proj": _unwrap(out_projs[i]).state_dict(),
        }, path / f"expert_{i}.pt")
    logger.info("Saved MoE checkpoint to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Joint fine-tune MoE (Router + Experts)")
    parser.add_argument("--expert-dir", type=str, default="./cluster_expert_checkpoints")
    parser.add_argument("--router-path", type=str, default="./router_checkpoints/final/router.pt")
    parser.add_argument("--cluster-labels", type=str, default="gt_clustering_results_3/cluster_labels.csv")
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for trajectory loss")
    parser.add_argument("--gumbel-tau", type=float, default=1.0, help="Gumbel-Softmax temperature")
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./moe_checkpoints")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=179)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
