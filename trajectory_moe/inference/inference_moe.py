# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目的轨迹 MoE 模块。
主要功能：基于 Alpamayo-R1 架构的 Macro-MoE 单个样本可视化推理入口。
此脚本在获取 MoE 预测轨迹后（利用 AlpamayoR1MoE 产生包含推理思维链 CoT 及 对应专家 Expert_idx），结合 cv2 和相机内参把预测轨迹特征投影绘制到原视角的驾驶图像中进行可视化展示。
注意：此脚本明确基于最新的 R1 版本架构。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo_r1.models.alpamayo_r1_moe`。
- 依赖原版 `inference.py` 中的绘图与截帧方法（如 `draw_trajectory_on_image`, `extract_frame_cv2`）。
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import torch

from alpamayo_r1.models.alpamayo_r1_moe import AlpamayoR1MoE
from alpamayo_r1 import helper
from inference import Config, build_dataset, draw_trajectory_on_image, extract_frame_cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpamayo-R1 MoE inference")
    parser.add_argument("--clip-id", type=str, default="5e18888d-03d7-4a56-b7c4-32492fb6b070")
    parser.add_argument("--t0-us", type=int, default=10_000_000)
    parser.add_argument("--model-dir", type=Path, default=Path("/path/to/models/Alpamayo-R1-10B"))
    parser.add_argument("--output-image", type=Path, default=Path("moe_trajectory.jpg"))
    parser.add_argument("--extracted-frame", type=Path, default=Path("frame_10s.jpg"))
    args = parser.parse_args()

    cfg = Config(
        clip_id=args.clip_id, t0_us=args.t0_us,
        model_dir=args.model_dir, output_image=args.output_image,
        extracted_frame=args.extracted_frame,
    )
    cfg.camera_name = "camera_front_wide_120fov"

    video_path = (
        cfg.video_root / cfg.camera_name
        / f"{cfg.clip_id}.{cfg.camera_name}.mp4"
    )
    extract_frame_cv2(video_path, cfg.t0_us, cfg.extracted_frame)

    data = build_dataset(cfg)

    logger.info("Loading MoE model from %s", cfg.model_dir)
    model = AlpamayoR1MoE.from_pretrained(str(cfg.model_dir), dtype=cfg.dtype).to(cfg.device)
    processor = helper.get_processor(model.tokenizer)

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

    torch.cuda.manual_seed_all(42)
    with torch.autocast(device_type=cfg.device, dtype=cfg.dtype):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True,
        )

    logger.info("Expert selection: %s", extra.get("expert_idx"))
    logger.info("CoT: %s", extra.get("cot", ["<none>"])[0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    min_ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1).min()
    logger.info("minADE: %.4f m", min_ade)

    draw_trajectory_on_image(
        cfg.extracted_frame, cfg.output_image, pred_xyz,
        data["intrinsics"], data["extrinsics"],
        data["intrinsics"]["width"], data["intrinsics"]["height"],
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
