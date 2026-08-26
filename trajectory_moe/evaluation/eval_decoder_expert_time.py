# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：评估聚合专家（Decoder Expert）在推理过程中的各阶段严格耗时（Timing）。
该脚本不仅计算 minADE 误差，还会利用计时器精确评估端到端流程：模型加载时间、视觉编码器(VisionEncoder)耗时、文本前缀填充(Prefilling)耗时以及轨迹采样解码(TrajDecoding)的单步耗时。评估 MoE 替换前后的速度收益。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `train_decoder_expert.py` 和 `alpamayo_r1` 中定义的底层网络设施。
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from alpamayo_r1 import helper
from train_decoder_expert import AlpamayoTrainDataset, collate_fn, build_model_with_light_expert

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def cuda_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _expand_cache(prefill_cache, num_traj_samples: int):
    from transformers import DynamicCache
    expanded = DynamicCache()
    for layer_idx, (k, v) in enumerate(prefill_cache):
        expanded.update(
            k.repeat_interleave(num_traj_samples, dim=0),
            v.repeat_interleave(num_traj_samples, dim=0),
            layer_idx,
        )
    return expanded


def sample_trajectories_from_prefill_cache(
    model, prefill_cache, hist_xyz, hist_rot, num_traj_samples: int, device: str, dtype
):
    B = hist_xyz.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    prefill_seq_len = prefill_cache.get_seq_length()

    expanded_cache = _expand_cache(prefill_cache, num_traj_samples)

    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = position_ids[None, None, :].expand(3, B * num_traj_samples, -1).clone()
    position_ids = position_ids + prefill_seq_len

    attention_mask = torch.zeros(
        (B * num_traj_samples, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=dtype, device=device,
    )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_bc = t[:, None, None] if t.dim() == 1 else t
        future_token_embeds = model.action_in_proj(x, t_bc)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=expanded_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        expanded_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

    sampled_action = model.diffusion.sample(
        batch_size=B * num_traj_samples,
        step_fn=step_fn,
        device=device,
        return_all_steps=False,
    )

    hist_xyz_rep = einops.repeat(hist_xyz, "b ... -> (b n) ...", n=num_traj_samples)
    hist_rot_rep = einops.repeat(hist_rot, "b ... -> (b n) ...", n=num_traj_samples)
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)
    pred_xyz = einops.rearrange(pred_xyz, "(b n) ... -> b 1 n ...", n=num_traj_samples)
    pred_rot = einops.rearrange(pred_rot, "(b n) ... -> b 1 n ...", n=num_traj_samples)
    return pred_xyz, pred_rot


def evaluate(args):
    device = args.device
    dtype = torch.bfloat16

    # --- Model loading time ---
    t_load_start = time.perf_counter()
    model = build_model_with_light_expert(args.base_model_dir, device, dtype)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.expert.load_state_dict(ckpt["expert"])
    model.action_in_proj.load_state_dict(ckpt["action_in_proj"])
    model.action_out_proj.load_state_dict(ckpt["action_out_proj"])
    model.eval()
    t_load_end = time.perf_counter()
    logger.info("Model loading time: %.3f s", t_load_end - t_load_start)
    logger.info("Loaded decoder expert from %s", args.checkpoint)

    processor = helper.get_processor(model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    eval_clips = clip_index[chunk_mask]
    clip_ids = eval_clips.index.tolist()
    logger.info("Evaluating on %d clips from chunks %d-%d", len(clip_ids), args.chunk_start, args.chunk_end)

    dataset = AlpamayoTrainDataset(clip_ids=clip_ids, t0_offsets_us=[6_000_000])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    results = []
    timing_records = []

    for i, batch in enumerate(dataloader):
        clip_id, t0_us = dataset.samples[i]
        chunk = eval_clips.loc[clip_id, "chunk"]

        hist_xyz = batch["ego_history_xyz"].to(device)
        hist_rot = batch["ego_history_rot"].to(device)

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
        gt_xyz = batch["ego_future_xyz"][0, 0, :, :2].numpy()

        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            fused_input_ids = model.fuse_traj_tokens(input_ids, traj_data)

            # --- Vision Encoder time ---
            # pixel_values processing happens inside vlm; we isolate it by running vision tower separately if possible,
            # otherwise we time the full vlm forward as Prefilling and note Vision Encoder is embedded within.
            # For models exposing visual_model / vision_tower, time it separately.
            vision_model = getattr(model.vlm, "visual", None) or getattr(model.vlm, "vision_tower", None)
            _ve_times: list[float] = []

            if vision_model is not None and "pixel_values" in vlm_kwargs:
                def _ve_pre_hook(module, args, kwargs):
                    _ve_times.append(cuda_sync_time())
                def _ve_post_hook(module, args, kwargs, output):
                    _ve_times.append(cuda_sync_time())
                h1 = vision_model.register_forward_pre_hook(_ve_pre_hook, with_kwargs=True)
                h2 = vision_model.register_forward_hook(_ve_post_hook, with_kwargs=True)

            # --- Prefilling time (includes Vision Encoder) ---
            t_prefill_start = cuda_sync_time()
            vlm_out = model.vlm(
                input_ids=fused_input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                **vlm_kwargs,
            )
            t_prefill_end = cuda_sync_time()
            t_prefill = t_prefill_end - t_prefill_start

            if vision_model is not None and "pixel_values" in vlm_kwargs:
                h1.remove()
                h2.remove()

            t_vision = (_ve_times[1] - _ve_times[0]) if len(_ve_times) == 2 else float("nan")

            prefill_cache = vlm_out.past_key_values

            # --- Trajectory Decoding time ---
            _decode_times: list[float] = []
            _orig_sample = model.diffusion.sample

            def _timed_sample(*a, **kw):
                _decode_times.append(cuda_sync_time())
                result = _orig_sample(*a, **kw)
                _decode_times.append(cuda_sync_time())
                return result

            model.diffusion.sample = _timed_sample
            pred_xyz, pred_rot = sample_trajectories_from_prefill_cache(
                model, prefill_cache, hist_xyz[:, 0], hist_rot[:, 0],
                args.num_traj_samples, device, dtype
            )
            model.diffusion.sample = _orig_sample
            t_decode = (_decode_times[1] - _decode_times[0]) if len(_decode_times) == 2 else float("nan")

        pred_xy = pred_xyz[0, 0, :, :, :2].cpu().numpy()
        min_ade = np.linalg.norm(pred_xy - gt_xyz[None], axis=-1).mean(axis=-1).min()

        results.append({"chunk": int(chunk), "clip_id": clip_id, "t0_us": t0_us, "min_ade": min_ade})
        timing_records.append({
            "clip_id": clip_id,
            "vision_encoder_s": t_vision,
            "prefilling_s": t_prefill,
            "traj_decoding_s": t_decode,
        })
        logger.info(
            "sample %d/%d | clip=%s | minADE %.4f | VisionEncoder=%.3fs Prefilling=%.3fs TrajDecoding=%.3fs",
            i + 1, len(dataset), clip_id, min_ade, t_vision, t_prefill, t_decode,
        )

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info("Results saved to %s | mean minADE = %.4f m", args.output, df["min_ade"].mean())

    df_time = pd.DataFrame(timing_records)
    time_output = args.output.replace(".csv", "_timing.csv")
    df_time.to_csv(time_output, index=False)
    logger.info(
        "Timing saved to %s | mean VisionEncoder=%.3fs Prefilling=%.3fs TrajDecoding=%.3fs",
        time_output,
        df_time["vision_encoder_s"].mean(),
        df_time["prefilling_s"].mean(),
        df_time["traj_decoding_s"].mean(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=str, default="/path/to/models/Alpamayo-R1-10B")
    parser.add_argument("--checkpoint", type=str, default="./decoder_expert_checkpoints/checkpoint-5000/decoder_expert.pt")
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=101)
    parser.add_argument("--chunk-end", type=int, default=102)
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_decoder_expert_time_results.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
