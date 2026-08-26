# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目，复现 Alpamayo-1.5 的端到端单帧可视化推理流程。
主要功能：实现 Alpamayo-1.5 的端到端推理，即输入多模态数据并预测轨迹，同时生成 BEV（鸟瞰图）可视化结果。
数据流与模块关系说明：
1. 【输入来源】：通过导入 `dataset.py` (PhysicalAIAVDatasetInterface) 获取原始视音频和自车运动轨迹数据；通过 `alpamayo1_5/helper.py` 辅助加载数据。
2. 【核心处理】：将格式整理好的张量(Tensor)输入给导入的核心模型 `alpamayo1_5/models/alpamayo1_5.py` 进行前向推理计算，并预测未来的轨迹、行为等。
3. 【输出去向】：产生的预测结果结合真实基准数据，传递给 `alpamayo1_5/viz_utils.py` 生成可视化的多阶段对比图表（如提取中间特征进行渲染等）。
"""

from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from torch import Tensor

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from alpamayo1_5.viz_utils import plot_bev_comparison, make_camera_grid
from dataset import PhysicalAIAVDatasetInterface

# 配置日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """推理流程的全局配置类"""
    clip_id: str = "275e7810-408a-43c7-84c0-f83dbf102268" # 测试片段的ID
    t0_us: int = 10_000_000  # 当前时刻（微秒）
    num_history_steps: int = 16 # 历史轨迹步数
    num_future_steps: int = 64  # 预测未来的轨迹步数
    time_step: float = 0.1      # 每步的时间间隔(秒)
    num_frames: int = 4         # 每个相机视角的历史帧数

    # 传感器外参文件路径
    extrinsics_parquet: Path = Path(
        "/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/sensor_extrinsics"
        "/sensor_extrinsics.chunk_0044.parquet"
    )
    # 传感器内参文件路径
    intrinsics_parquet: Path = Path(
        "/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/camera_intrinsics"
        "/camera_intrinsics.chunk_0044.parquet"
    )
    # 预训练模型路径
    model_dir: Path = Path("/path/to/data/Alpamayo-1.5-10B")
    video_root: Path = Path("/path/to/data/PhysicalAI-Autonomous-Vehicles/camera")
    output_image: Path = Path("bev_trajectory.png")  # 渲染输出的预测轨迹图

    # 语言导航指令
    nav_text: str = "Continue straight" # 正确的导航文本
    nav_text_swapped: str = "Turn left" # 设定的反事实（对比）导航文本

    verbose: bool = True
    device: str = "cuda" # 运行设备
    dtype: torch.dtype = torch.bfloat16 # 推理数据类型


def quat_to_rot(x, y, z, w):
    """将四元数转换为3x3旋转矩阵"""
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_extrinsics(parquet_path, clip_id, sensor_name):
    """从parquet文件中读取指定传感器的外参（四元数和平移向量）"""
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, sensor_name)]
    qx, qy, qz, qw = row[["qx", "qy", "qz", "qw"]].astype(float).values
    tx, ty, tz = row[["x", "y", "z"]].astype(float).values
    return np.array([qx, qy, qz, qw], dtype=np.float64), np.array([tx, ty, tz], dtype=np.float64)


def load_intrinsics(parquet_path, clip_id, camera_name):
    df = pd.read_parquet(parquet_path)
    row = df.loc[(clip_id, camera_name)]
    width, height, cx, cy = row[["width", "height", "cx", "cy"]].astype(float).values
    fw_coeffs = row[["fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4"]].astype(float).values
    return int(width), int(height), float(cx), float(cy), fw_coeffs[::-1]


def build_dataset(cfg):
    avdi = PhysicalAIAVDatasetInterface()
    sensor_name = cfg.camera_name
    q, t = load_extrinsics(cfg.extrinsics_parquet, cfg.clip_id, sensor_name)
    R_wc = quat_to_rot(*q)
    t_wc = t

    width, height, cx, cy, fw_poly = load_intrinsics(cfg.intrinsics_parquet, cfg.clip_id, sensor_name)

    egomotion = avdi.get_clip_feature(cfg.clip_id, avdi.features.LABELS.EGOMOTION, types="egomotion")

    assert cfg.t0_us > cfg.num_history_steps * cfg.time_step * 1_000_000

    history_offsets_us = np.arange(
        -(cfg.num_history_steps - 1) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000 / 2,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = cfg.t0_us + history_offsets_us

    future_offsets_us = np.arange(
        cfg.time_step * 1_000_000,
        (cfg.num_future_steps + 0.5) * cfg.time_step * 1_000_000,
        cfg.time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = cfg.t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    image_timestamps = np.array([
        cfg.t0_us - (cfg.num_frames - 1 - i) * int(cfg.time_step * 1_000_000)
        for i in range(cfg.num_frames)
    ], dtype=np.int64)

    image_frames_list: List[Tensor] = []
    camera_indices_list: List[int] = []
    timestamps_list: List[Tensor] = []

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(cfg.clip_id, cam_feature, types="camera")
        frames_np, frame_ts = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames_np)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        cam_name = cam_feature.split("/")[-1].lower() if isinstance(cam_feature, str) else ""
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_ts.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": cfg.t0_us,
        "clip_id": cfg.clip_id,
    }


def run_model(model, processor, data, device, dtype, nav_text=None):
    """
    运行视觉-语言-动作模型的推理。
    1. 根据图像帧、相机序号和可选的导航文本创建输入消息。
    2. 使用处理器(Processor)编码数据，得到模型能够接受的Token格式。
    3. 加入之前的历史位姿信息（XYZ平移和旋转矩阵）。
    4. 执行模型采样：在自回归生成的文本中预测控制动作以及未来的轨迹点。
    """
    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
        nav_text=nav_text, # 导航文本作为条件注入
    )
    # 利用预置Prompt模板和分词器处理图像与文本
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    # 组合送入Alpamayo模型的数据包
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    torch.cuda.manual_seed_all(42) # 保证多次实验结果可重复
    
    # 开启半精度推理以节省显存加速推理
    with torch.autocast(device_type=device, dtype=dtype):
        # 采样多模态大模型的输出（包含预测轨迹XYZ/旋转，以及一些自回归产生的其他Token数据）
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True, # 最大生成Token长为256
        )
    return pred_xyz, pred_rot, extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpamayo-1.5 trajectory prediction with BEV visualization.")
    parser.add_argument("--clip-id", type=str, default="275e7810-408a-43c7-84c0-f83dbf102268")
    parser.add_argument("--t0-us", type=int, default=10_000_000)
    parser.add_argument("--model-dir", type=Path, default=Path("/path/to/data/Alpamayo-1.5-10B"))
    parser.add_argument("--output-image", type=Path, default=Path("bev_trajectory.png"))
    parser.add_argument("--nav-text", type=str, default="Continue straight")
    parser.add_argument("--nav-text-swapped", type=str, default="Turn left")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        clip_id=args.clip_id, t0_us=args.t0_us, model_dir=args.model_dir,
        output_image=args.output_image, nav_text=args.nav_text,
        nav_text_swapped=args.nav_text_swapped, verbose=args.verbose,
    )
    cfg.camera_name = "camera_front_wide_120fov"

    data = build_dataset(cfg)

    logger.info("Loading Alpamayo-1.5 model from %s", cfg.model_dir)
    model = Alpamayo1_5.from_pretrained(str(cfg.model_dir), dtype=cfg.dtype).to(cfg.device)
    processor = helper.get_processor(model.tokenizer)

    # Run with nav, without nav, and with swapped nav
    # 执行包含正常导航指令的预测
    pred_with_nav, _, _ = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=cfg.nav_text)
    # 执行不含导航指令的裸预测
    pred_no_nav, _, extra = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=None)
    # 执行含反事实(Swapped)或错误指令的预测（测试指令服从性）
    pred_counterfactual, _, _ = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=cfg.nav_text_swapped)

    logger.info("Inference completed.")
    print("extra: ", extra)

    # 制作多相机视角拼接的网格（用于在渲染结果中展示当时看到了什么）
    camera_grid = make_camera_grid(data["image_frames"], data["camera_indices"])

    # 利用渲染工具(viz_utils)，在鸟瞰平面图(BEV)上同时绘制：当前指令轨迹、无指令轨迹、反向指令轨迹、GT真实历史与未来轨迹
    fig = plot_bev_comparison(
        pred_with_nav=pred_with_nav,
        pred_no_nav=pred_no_nav,
        pred_counterfactual=pred_counterfactual,
        nav_text=cfg.nav_text,
        nav_text_swapped=cfg.nav_text_swapped,
        gt_future_xyz=data["ego_future_xyz"],
        camera_images=camera_grid,
        title=f"BEV Trajectory Comparison - Clip: {cfg.clip_id}",
    )
    # 将matplotlib图保存到磁盘
    fig.savefig(str(cfg.output_image), dpi=150, bbox_inches="tight")
    logger.info("BEV visualization saved to %s", cfg.output_image)

    logger.info("All done!")


if __name__ == "__main__":
    main()
