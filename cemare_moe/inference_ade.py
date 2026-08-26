# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目，输出 Alpamayo1_5 的 ADE 的单帧推理结果。
主要功能：针对带有 1.5 评估或配置的端到端推理，即输入多模态数据并预测轨迹，计算 ADE 指标，并可选择性生成 BEV（鸟瞰图）可视化结果。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `dataset.py`：使用里面的 `PhysicalAIAVDatasetInterface` 来加载原始的图像和自车运动轨迹数据。
- 依赖 `alpamayo1_5/helper.py`：调用它来组装给模型的输入格式、加载处理器(Processor)等辅助操作。
- 依赖 `alpamayo1_5/models/alpamayo1_5.py`：使用定义好的 Alpamayo 核心模型来做实际的推理计算。
- 依赖 `alpamayo1_5/viz_utils.py`：如果开启可视化，用来画出对比的 BEV 轨迹图。

向下提供（谁调它）：
- 这是个入口脚本，通常直接在终端运行。也可以把里面的结果或者方法提供给其他评测脚本用。
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
    nav_text: str = None # 默认无导航
    nav_text_swapped: str = None # 默认无对照导航

    verbose: bool = True
    device: str = "cuda" # 运行设备
    dtype: torch.dtype = torch.bfloat16 # 推理数据类型
    visualize: bool = False # 是否生成可视化图片
    cameras_config: str = "2_cam"  # 摄像头配置: "1_cam", "2_cam", "4_cam"

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

    # 动态构建摄像头配置
    if cfg.cameras_config == "1_cam":
        camera_features = [
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        ]
    elif cfg.cameras_config == "3_cam":
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        ]
    elif cfg.cameras_config == "cross_front_rl":
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_REAR_LEFT_70FOV,
        ]    
    elif cfg.cameras_config == "cross_front_rr":
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_REAR_RIGHT_70FOV,
        ]    
    elif cfg.cameras_config == "cross_front_rt":
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_REAR_TELE_30FOV,
        ]
    elif cfg.cameras_config == "7_cam":
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_REAR_LEFT_70FOV,
            avdi.features.CAMERA.CAMERA_REAR_TELE_30FOV,
            avdi.features.CAMERA.CAMERA_REAR_RIGHT_70FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]
    else:  # default 4_cam
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    # 统一构建并打印允许加载的摄像头列表，供底层使用以避免额外下载
    allowed_camera_features = [cf if isinstance(cf, str) else cf for cf in camera_features]
    logger.info("Camera config requested: %s", cfg.cameras_config)
    logger.info("Camera features to load: %s", allowed_camera_features)

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
        # 传递 allowed_camera_features 给底层数据接口，避免不必要的数据访问/下载
        camera = avdi.get_clip_feature(
            cfg.clip_id,
            cam_feature,
            types="camera",
            allowed_camera_features=allowed_camera_features,
        )
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


def run_model(model, processor, data, device, dtype, nav_text=None, num_traj_samples=6):
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
            num_traj_samples=num_traj_samples, max_generation_length=256, return_extra=True, # 最大生成Token长为256
        )
    return pred_xyz, pred_rot, extra


def compute_min_ade_6(pred_trajs, gt_traj):
    """
    计算 minADE_6
    
    Args:
        pred_trajs: 模型预测的 6 条轨迹，shape 假设为 [batch_size, 6, num_timesteps, 3] (包含 x, y, yaw)
        gt_traj: 真实的 1 条轨迹，shape 为 [batch_size, num_timesteps, 3] (包含 x, y, yaw)
        
    Returns:
        min_ade: 每个 batch 样本的 minADE_6 得分，shape [batch_size]
    """
    pred_trajs = np.asarray(pred_trajs)
    gt_traj = np.asarray(gt_traj)
    
    if gt_traj.ndim == 4: # [1, 1, 64, 3] -> [1, 64, 3]
        gt_traj = gt_traj.squeeze(axis=1)
    
    # Handle possible extra dimensions in pred_trajs
    if pred_trajs.ndim > 4:
        pred_trajs = np.squeeze(pred_trajs, axis=tuple(range(1, pred_trajs.ndim - 3)))
        
    batch_size = pred_trajs.shape[0]
    num_modes = pred_trajs.shape[1] # 这里应该是 6
    
    # 【重点1】：剔除 z 轴或 yaw 角，只保留 2D 平面的 (x, y)
    pos_gt = gt_traj[..., :2]         # shape: [batch_size, num_timesteps, 2]
    
    all_modes_ade = []
    
    # 保证预测长度和真实长度一致对齐进行计算
    num_timesteps = min(pred_trajs.shape[2], pos_gt.shape[1])
    pos_gt = pos_gt[:, :num_timesteps, :]
    
    # 遍历这 6 条候选轨迹
    for i in range(num_modes):
        pos_pred = pred_trajs[:, i, :num_timesteps, :2]  # 提取第 i 条轨迹的 (x, y)
        
        # 计算欧氏距离
        distances = np.linalg.norm(pos_pred - pos_gt, axis=-1) # shape: [batch_size, num_timesteps]
        
        # 【重点2：求平均】：在时间步维度 (axis=1) 上求平均，得到单条轨迹的 ADE
        ade = np.mean(distances, axis=1) # shape: [batch_size]
        all_modes_ade.append(ade)
        
    # 将 6 个 ADE 拼接到一起
    all_modes_ade = np.stack(all_modes_ade, axis=1) # shape: [batch_size, 6]
    
    # 【重点3：求最小】：在 6 个模态维度 (axis=1) 上取最小值，得到 minADE_6
    min_ade = np.min(all_modes_ade, axis=1) # shape: [batch_size]
    
    return min_ade

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpamayo-1.5 trajectory prediction with ADE score and optional BEV visualization.")
    parser.add_argument("--clip-id", type=str, default="275e7810-408a-43c7-84c0-f83dbf102268")
    parser.add_argument("--t0-us", type=int, default=10_000_000)
    parser.add_argument("--model-dir", type=Path, default=Path("/path/to/data/Alpamayo-1.5-10B"))
    parser.add_argument("--output-image", type=Path, default=Path("bev_trajectory.png"))
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--nav-text-swapped", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    # 增加摄像头配置参数
    parser.add_argument("--cameras-config", type=str, default=None, choices=["1_cam", "3_cam", "4_cam", "cross_front_rl", "cross_front_rr", "cross_front_rt", "7_cam"], help="Camera configuration to use")
    # 增加可视化选项，默认为不生成图片，使用 --visualize 才会渲染并保存图片
    parser.add_argument("--visualize", action="store_true", help="Generate and save BEV trajectory image")
    args = parser.parse_args()

    cfg = Config(
        clip_id=args.clip_id, t0_us=args.t0_us, model_dir=args.model_dir,
        output_image=args.output_image, nav_text=args.nav_text,
        nav_text_swapped=args.nav_text_swapped, verbose=args.verbose,
        visualize=args.visualize
    )
    # Only override dataclass default if user explicitly provided the arg
    if args.cameras_config is not None:
        cfg.cameras_config = args.cameras_config
    cfg.camera_name = "camera_front_wide_120fov"

    logger.info("Initializing dataset with clip_id: %s and config: %s", cfg.clip_id, cfg.cameras_config)
    data = build_dataset(cfg)
    
    # 打印加载成功了哪些摄像头视野
    loaded_cam_indices = data["camera_indices"].tolist()
    index_to_camera_name = {
        0: "camera_cross_left_120fov",
        1: "camera_front_wide_120fov",
        2: "camera_cross_right_120fov",
        6: "camera_front_tele_30fov"
    }
    loaded_cam_names = [index_to_camera_name.get(idx, "unknown_cam") for idx in loaded_cam_indices]
    logger.info("==> Successfully loaded %d camera(s): %s", len(loaded_cam_names), ", ".join(loaded_cam_names))

    logger.info("Loading Alpamayo-1.5 model from %s", cfg.model_dir)
    model = Alpamayo1_5.from_pretrained(str(cfg.model_dir), dtype=cfg.dtype).to(cfg.device)
    processor = helper.get_processor(model.tokenizer)

    # 真实基准轨迹 (GT)
    gt_future_xyz = data["ego_future_xyz"]

    if cfg.nav_text is not None:
        # 执行包含正常导航指令的预测
        logger.info("Running prediction with correct navigation text...")
        pred_with_nav, _, _ = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=cfg.nav_text)
        ade_with_nav = compute_min_ade_6(pred_with_nav.cpu().numpy(), gt_future_xyz.cpu().numpy())
        logger.info(f"ADE (with correct nav): {ade_with_nav} meters")
    else:
        pred_with_nav = None

    # 根据可视化或对照需求，计算其他情况的预测
    pred_no_nav = None
    pred_counterfactual = None

    # 执行不含导航指令的裸预测
    logger.info("Running prediction without navigation text...")
    pred_no_nav, _, extra = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=None)
    ade_no_nav = compute_min_ade_6(pred_no_nav.cpu().numpy(), gt_future_xyz.cpu().numpy())
    logger.info(f"ADE (without nav): {ade_no_nav} meters")

    if cfg.visualize or cfg.verbose:
        if cfg.nav_text_swapped is not None:
            # 执行含反事实(Swapped)或错误指令的预测（测试指令服从性）
            logger.info("Running prediction with swapped navigation text...")
            pred_counterfactual, _, _ = run_model(model, processor, data, cfg.device, cfg.dtype, nav_text=cfg.nav_text_swapped)
            ade_counterfactual = compute_min_ade_6(pred_counterfactual.cpu().numpy(), gt_future_xyz.cpu().numpy())
            logger.info(f"ADE (with swapped nav): {ade_counterfactual} meters")
        
        if cfg.verbose:
            print("extra info from no_nav run: ", extra)

    logger.info("Inference completed.")

    # 可选：渲染并保存 BEV 可视化图
    if cfg.visualize:
        logger.info("Generating BEV visualization...")
        # 制作多相机视角拼接的网格（用于在渲染结果中展示当时看到了什么）
        camera_grid = make_camera_grid(data["image_frames"], data["camera_indices"])

        # 利用渲染工具(viz_utils)，在鸟瞰平面图(BEV)上同时绘制：当前指令轨迹、无指令轨迹、反向指令轨迹、GT真实历史与未来轨迹
        fig = plot_bev_comparison(
            pred_with_nav=pred_with_nav,
            pred_no_nav=pred_no_nav,
            pred_counterfactual=pred_counterfactual,
            nav_text=cfg.nav_text,
            nav_text_swapped=cfg.nav_text_swapped,
            gt_future_xyz=gt_future_xyz,
            camera_images=camera_grid,
            title=f"BEV Trajectory Comparison - Clip: {cfg.clip_id}",
        )
        # 将matplotlib图保存到磁盘
        fig.savefig(str(cfg.output_image), dpi=150, bbox_inches="tight")
        logger.info("BEV visualization saved to %s", cfg.output_image)

    logger.info("All done!")


if __name__ == "__main__":
    main()
