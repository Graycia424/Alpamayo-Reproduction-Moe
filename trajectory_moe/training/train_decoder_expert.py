# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构，通过提取 Prefill 阶段（思维链 CoC 生成前）的 KV Cache 进行轻量级专家（Decoder Expert）训练（单卡版本）。
它使用 Flow-Matching 算法专门对下游的 Action 解码进行加速适配（轻量化网络配置定义在脚本上方）。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 完全依赖 `alpamayo_r1.models.alpamayo_r1` 大一统基座。
"""

from __future__ import annotations
import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from dataset import PhysicalAIAVDatasetInterface

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight expert config (same as train_light_expert.py)
# ---------------------------------------------------------------------------
LIGHT_EXPERT_CFG = {
    "dtype": "bfloat16",
    "hidden_size": 3584,           # 保持原始宽度
    "num_hidden_layers": 8,        # 只减层数 28 -> 8
    "intermediate_size": 5120,     # 保持原始中间层
}

LIGHT_ACTION_IN_PROJ_CFG = {
    "_target_": "alpamayo_r1.models.action_in_proj.PerWaypointActionInProjV2",
    "hidden_size": 1024,
    "max_freq": 30.0,
    "num_enc_layers": 1,
    "num_fourier_feats": 8,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AlpamayoTrainDataset(Dataset):
    def __init__(
        self,
        clip_ids: list[str],
        t0_offsets_us: list[int] | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        num_frames: int = 4,
    ):
        self.avdi = PhysicalAIAVDatasetInterface()
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.num_frames = num_frames
        self.samples: list[tuple[str, int]] = []
        for cid in clip_ids:
            if t0_offsets_us is not None:
                for t0 in t0_offsets_us:
                    self.samples.append((cid, t0))
            else:
                for t0 in range(2_000_000, 12_000_000, 1_000_000):
                    self.samples.append((cid, t0))
        self.camera_features = [
            self.avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            self.avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            self.avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            self.avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id, t0_us = self.samples[idx]
        dt_us = int(self.time_step * 1_000_000)
        egomotion = self.avdi.get_clip_feature(
            clip_id, self.avdi.features.LABELS.EGOMOTION, types="egomotion"
        )
        history_ts = t0_us + np.arange(
            -(self.num_history_steps - 1) * dt_us, dt_us // 2, dt_us
        ).astype(np.int64)
        future_ts = t0_us + np.arange(
            dt_us, int((self.num_future_steps + 0.5) * dt_us), dt_us
        ).astype(np.int64)
        ego_hist = egomotion(history_ts)
        ego_fut = egomotion(future_ts)
        hist_xyz = ego_hist.pose.translation
        hist_quat = ego_hist.pose.rotation.as_quat()
        fut_xyz = ego_fut.pose.translation
        fut_quat = ego_fut.pose.rotation.as_quat()
        t0_xyz = hist_xyz[-1].copy()
        t0_rot_inv = spt.Rotation.from_quat(hist_quat[-1]).inv()
        hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
        fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
        hist_rot_local = (t0_rot_inv * spt.Rotation.from_quat(hist_quat)).as_matrix()
        fut_rot_local = (t0_rot_inv * spt.Rotation.from_quat(fut_quat)).as_matrix()
        image_ts = np.array(
            [t0_us - (self.num_frames - 1 - i) * dt_us for i in range(self.num_frames)],
            dtype=np.int64,
        )
        frames_list = []
        for cam_feat in self.camera_features:
            cam = self.avdi.get_clip_feature(clip_id, cam_feat, types="camera")
            frames_np, _ = cam.decode_images_from_timestamps(image_ts)
            frames_list.append(torch.from_numpy(frames_np))
        all_frames = torch.stack(frames_list, dim=0)
        all_frames = rearrange(all_frames, "n t h w c -> (n t) c h w")
        return {
            "image_frames": all_frames,
            "ego_history_xyz": torch.from_numpy(hist_xyz_local).float().unsqueeze(0),
            "ego_history_rot": torch.from_numpy(hist_rot_local).float().unsqueeze(0),
            "ego_future_xyz": torch.from_numpy(fut_xyz_local).float().unsqueeze(0),
            "ego_future_rot": torch.from_numpy(fut_rot_local).float().unsqueeze(0),
        }


def collate_fn(batch):
    return {
        "image_frames": [b["image_frames"] for b in batch],
        "ego_history_xyz": torch.stack([b["ego_history_xyz"] for b in batch]),
        "ego_history_rot": torch.stack([b["ego_history_rot"] for b in batch]),
        "ego_future_xyz": torch.stack([b["ego_future_xyz"] for b in batch]),
        "ego_future_rot": torch.stack([b["ego_future_rot"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Build model with lightweight expert
# ---------------------------------------------------------------------------
def build_model_with_light_expert(model_dir: str, device: str, dtype: torch.dtype) -> AlpamayoR1:
    """Load pretrained AlpamayoR1, replace expert/in_proj/out_proj with lightweight versions."""
    import hydra.utils as hyu

    model = AlpamayoR1.from_pretrained(model_dir, torch_dtype=dtype).to(device=device, dtype=dtype)

    config = model.config
    config.expert_cfg = {**config.expert_cfg, **LIGHT_EXPERT_CFG}
    config.action_in_proj_cfg = LIGHT_ACTION_IN_PROJ_CFG

    expert_text_config = copy.deepcopy(model.vlm.config.text_config)
    for k, v in LIGHT_EXPERT_CFG.items():
        if k != "dtype":
            setattr(expert_text_config, k, v)

    new_expert = AutoModel.from_config(expert_text_config).to(device=device, dtype=dtype)
    del new_expert.embed_tokens

    new_in_proj = hyu.instantiate(
        LIGHT_ACTION_IN_PROJ_CFG,
        in_dims=model.action_space.get_action_space_dims(),
        out_dim=1024,
    ).to(device=device, dtype=dtype)

    new_out_proj = torch.nn.Linear(1024, model.action_space.get_action_space_dims()[-1]).to(
        device=device, dtype=dtype
    )

    model.expert = new_expert
    model.action_in_proj = new_in_proj
    model.action_out_proj = new_out_proj

    for p in model.parameters():
        p.requires_grad = False
    for m in [model.expert, model.action_in_proj, model.action_out_proj]:
        for p in m.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %d total, %d trainable (%.2f%%)", total, trainable, 100 * trainable / total)
    return model


# ---------------------------------------------------------------------------
# Training loop - uses KV cache BEFORE CoC generation
# ---------------------------------------------------------------------------
def train(args):
    device = args.device
    dtype = torch.bfloat16

    model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    processor = helper.get_processor(model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()
    logger.info("Loaded %d clips from chunks %d-%d", len(clip_ids), args.chunk_start, args.chunk_end)

    dataset = AlpamayoTrainDataset(clip_ids=clip_ids)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
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

            gt_actions = model.action_space.traj_to_action(
                traj_history_xyz=hist_xyz[:, 0],
                traj_history_rot=hist_rot[:, 0],
                traj_future_xyz=fut_xyz[:, 0],
                traj_future_rot=fut_rot[:, 0],
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
                # KEY DIFFERENCE: Get KV cache from VLM prefill BEFORE any CoC generation
                # This is the raw visual+text understanding without chain-of-thought reasoning
                fused_input_ids = model.fuse_traj_tokens(input_ids, traj_data)
                with torch.no_grad():
                    vlm_out = model.vlm(
                        input_ids=fused_input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_hidden_states=False,
                        **vlm_kwargs,
                    )
                # This is the prefill KV cache - BEFORE any autoregressive CoC generation
                prefill_cache = vlm_out.past_key_values

                n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
                loss = _compute_fm_loss(
                    model, gt_actions, prefill_cache, n_diffusion_tokens, device, dtype
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            del prefill_cache

            if step % args.log_every == 0:
                logger.info("step %d | fm_loss %.4f", step, loss.item())

            if step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"checkpoint-{step}"
                _save_decoder_expert(model, save_path)
                logger.info("Saved checkpoint to %s", save_path)

    final_path = Path(args.output_dir) / "final"
    _save_decoder_expert(model, final_path)
    logger.info("Training complete. Final model saved to %s", final_path)


def _compute_fm_loss(
    model: AlpamayoR1,
    gt_actions: torch.Tensor,
    prefill_cache,
    n_diffusion_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute flow-matching loss using prefill KV cache (before CoC)."""
    B = gt_actions.shape[0]

    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(gt_actions)
    t_bc = t[:, None, None]
    x_t = (1 - t_bc) * noise + t_bc * gt_actions
    target = gt_actions - noise

    prefill_seq_len = prefill_cache.get_seq_length()
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = position_ids[None, None, :].expand(3, B, -1).clone()
    position_ids = position_ids + prefill_seq_len

    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=dtype,
        device=device,
    )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    future_token_embeds = model.action_in_proj(x_t, t_bc)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(B, n_diffusion_tokens, -1)

    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prefill_cache,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    prefill_cache.crop(prefill_seq_len)

    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred = model.action_out_proj(last_hidden).view(B, *model.action_space.get_action_space_dims())

    return torch.nn.functional.mse_loss(pred, target)


def _save_decoder_expert(model: AlpamayoR1, path: Path):
    """Save only the decoder expert modules."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert": model.expert.state_dict(),
            "action_in_proj": model.action_in_proj.state_dict(),
            "action_out_proj": model.action_out_proj.state_dict(),
        },
        path / "decoder_expert.pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Train trajectory decoder expert (pre-CoC)")
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--output-dir", type=str, default="./decoder_expert_checkpoints_chunk200")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
