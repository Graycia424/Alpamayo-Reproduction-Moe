# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：对基线的 Alpamayo-R1 大模型架构进行极限的内部耗时（Latency）测定。
利用 PyTorch 的 Hook 机制强行插入网络最内部，精确记录：1. 视觉特征提取 (Vision Encoder); 2. 文本词元处理 (Prefilling); 3. 思维链自回归生成 (CoC generation); 4. 最终扩散采样 (Trajectory Decoding)。
为论文或实验说明（MoE如何减轻大模型巨大延迟）提供强力的数据佐证。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo_r1.models.alpamayo_r1` 与 `alpamayo_r1.helper`。
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from train_decoder_expert import AlpamayoTrainDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def cuda_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def evaluate(args):
    device = args.device
    dtype = torch.bfloat16

    # --- Model loading time ---
    t_load_start = time.perf_counter()
    model = AlpamayoR1.from_pretrained(str(args.model_dir), dtype=dtype).to(device)
    model.eval()
    t_load_end = time.perf_counter()
    logger.info("Model loading time: %.3f s", t_load_end - t_load_start)

    processor = helper.get_processor(model.tokenizer)

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    eval_clips = clip_index[chunk_mask]
    clip_ids = eval_clips.index.tolist()
    logger.info("Evaluating on %d clips from chunks %d-%d", len(clip_ids), args.chunk_start, args.chunk_end)

    dataset = AlpamayoTrainDataset(clip_ids=clip_ids, t0_offsets_us=[6_000_000])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Locate vision model for hook-based timing
    vision_model = getattr(model.vlm, "visual", None) or getattr(model.vlm, "vision_tower", None)

    results = []
    timing_records = []

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

        # --- Hook: Vision Encoder ---
        _ve_times: list[float] = []
        hooks = []
        if vision_model is not None:
            hooks.append(vision_model.register_forward_pre_hook(
                lambda m, a, k: _ve_times.append(cuda_sync_time()), with_kwargs=True
            ))
            hooks.append(vision_model.register_forward_hook(
                lambda m, a, k, o: _ve_times.append(cuda_sync_time()), with_kwargs=True
            ))

        # --- Hook: VLM forward call counter to split Prefilling vs CoC decode ---
        # First call inside generate() = prefilling; subsequent calls = CoC token generation
        _vlm_forward_times: list[float] = []  # [pre0, post0, pre1, post1, ...]

        def _vlm_pre(m, a, k):
            _vlm_forward_times.append(cuda_sync_time())

        def _vlm_post(m, a, k, o):
            _vlm_forward_times.append(cuda_sync_time())

        hooks.append(model.vlm.register_forward_pre_hook(_vlm_pre, with_kwargs=True))
        hooks.append(model.vlm.register_forward_hook(_vlm_post, with_kwargs=True))

        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            # Patch diffusion.sample to capture trajectory decoding time
            _decode_times: list[float] = []
            _orig_sample = model.diffusion.sample

            def _timed_sample(*a, **kw):
                _decode_times.append(cuda_sync_time())
                result = _orig_sample(*a, **kw)
                _decode_times.append(cuda_sync_time())
                return result

            model.diffusion.sample = _timed_sample

            t_infer_start = cuda_sync_time()
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                num_traj_samples=args.num_traj_samples,
                return_extra=True,
            )
            t_infer_end = cuda_sync_time()

            model.diffusion.sample = _orig_sample

        for h in hooks:
            h.remove()

        t_vision = (_ve_times[1] - _ve_times[0]) if len(_ve_times) == 2 else float("nan")
        t_decode = (_decode_times[1] - _decode_times[0]) if len(_decode_times) == 2 else float("nan")

        # _vlm_forward_times: pairs of (pre, post) per forward call
        # call 0 = prefilling, calls 1..N = CoC token generation
        if len(_vlm_forward_times) >= 2:
            t_prefill = _vlm_forward_times[1] - _vlm_forward_times[0]
            t_coc = (_vlm_forward_times[-1] - _vlm_forward_times[2]) if len(_vlm_forward_times) >= 4 else 0.0
        else:
            t_prefill = float("nan")
            t_coc = float("nan")

        pred_xy = pred_xyz[0, 0, :, :, :2].cpu().numpy()
        min_ade = np.linalg.norm(pred_xy - gt_xyz[None], axis=-1).mean(axis=-1).min()
        coc = extra.get("cot", [[[""]]])[0][0][0] if "cot" in extra else ""

        t_total = t_infer_end - t_infer_start

        results.append({"chunk": int(chunk), "clip_id": clip_id, "t0_us": t0_us, "min_ade": min_ade, "coc": coc})
        timing_records.append({
            "clip_id": clip_id,
            "total_inference_s": t_total,
            "vision_encoder_s": t_vision,
            "prefilling_s": t_prefill,
            "coc_generation_s": t_coc,
            "traj_decoding_s": t_decode,
        })
        logger.info(
            "sample %d/%d | clip=%s | minADE %.4f | Total=%.3fs VisionEncoder=%.3fs Prefilling=%.3fs CoC=%.3fs TrajDecoding=%.3fs",
            i + 1, len(dataset), clip_id, min_ade, t_total, t_vision, t_prefill, t_coc, t_decode,
        )
        print(f"CoC: {coc}\n")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info("Results saved to %s | mean minADE = %.4f m", args.output, df["min_ade"].mean())

    df_time = pd.DataFrame(timing_records)
    time_output = args.output.replace(".csv", "_timing.csv")
    df_time.to_csv(time_output, index=False)
    logger.info(
        "Timing saved to %s | mean Total=%.3fs VisionEncoder=%.3fs Prefilling=%.3fs CoC=%.3fs TrajDecoding=%.3fs",
        time_output,
        df_time["total_inference_s"].mean(),
        df_time["vision_encoder_s"].mean(),
        df_time["prefilling_s"].mean(),
        df_time["coc_generation_s"].mean(),
        df_time["traj_decoding_s"].mean(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/path/to/models/Alpamayo-R1-10B"))
    parser.add_argument("--clip-index", type=str,
                        default="/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=101)
    parser.add_argument("--chunk-end", type=int, default=102)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_samples_inference_time_results.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
