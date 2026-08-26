# -*- coding: utf-8 -*-
"""
快速轨迹标签提取脚本。
不加载大模型，只读取数据集的未来真实轨迹 (Ground Truth)，
根据车辆在未来数秒的 Y 轴（横向）偏移量，自动判定该场景是直行还是转弯。
用数据集的简单场景分布来为每个 Clip 打标签，方便后续分析不同场景下模型表现的差异。
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 从已有的文件中复用配置和数据集读取逻辑
from inference_ade import Config, build_dataset
from run_multiple_clips import load_clip_index_local

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    # 保持跟测试脚本用一样的参数，方便对齐数据
    p.add_argument("--clip-index", type=Path, default=Path("/path/to/data/clip_index.parquet"))
    p.add_argument("--chunk-ids", type=str, default="10,20,30,40,50,60,70,80,90,100")
    p.add_argument("--num-clips", type=int, default=100)
    p.add_argument("--output-csv", type=Path, default=Path("clip_labels.csv"))
    # 为了跑得最快，只请求1个最基本的摄像头
    p.add_argument("--cameras-config", type=str, default="1_cam") 
    p.add_argument("--device", type=str, default="cpu", help="使用CPU即可，不需要跑模型")
    return p.parse_args()

def main():
    args = parse_args()
    clip_list = []
    
    # 1. 像之前一样获取需要跑的所有 clip_id
    chunk_ids = [int(c.strip()) for c in args.chunk_ids.split(",") if c.strip()]
    for cid in chunk_ids:
        cls = load_clip_index_local(args.clip_index, cid, args.num_clips)
        clip_list.extend(cls)
        
    if not clip_list:
        logger.error("没有找到任何 Clip！")
        return

    logger.info(f"开始提取标签，共计 {len(clip_list)} 个 Clips。结果将保存至 {args.output_csv}")
    
    # 初始化输出文件
    with open(args.output_csv, 'w', encoding='utf-8') as f:
        f.write("clip_id,final_x_offset,final_y_offset,scenario\n")

    base_cfg = Config(cameras_config=args.cameras_config)
    
    # 2. 循环读取轨迹并打标签
    for i, item in enumerate(clip_list, start=1):
        if isinstance(item, tuple):
            clip_id, chunk_id = item
            chunk_str = f"{chunk_id:04d}"
            base_cfg.extrinsics_parquet = Path(f"/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/sensor_extrinsics/sensor_extrinsics.chunk_{chunk_str}.parquet")
            base_cfg.intrinsics_parquet = Path(f"/path/to/data/PhysicalAI-Autonomous-Vehicles/calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk_str}.parquet")
        else:
            clip_id = item
        
        cfg = Config(
            clip_id=clip_id,
            t0_us=base_cfg.t0_us,
            num_history_steps=base_cfg.num_history_steps,
            num_future_steps=base_cfg.num_future_steps,
            time_step=base_cfg.time_step,
            num_frames=base_cfg.num_frames,
            extrinsics_parquet=base_cfg.extrinsics_parquet,
            intrinsics_parquet=base_cfg.intrinsics_parquet,
            model_dir=base_cfg.model_dir,
            video_root=base_cfg.video_root,
            device=args.device,
        )
        cfg.cameras_config = args.cameras_config
        cfg.camera_name = "camera_front_wide_120fov"

        try:
            # 只加载数据，不跑大模型推理！速度极快
            data = build_dataset(cfg)
            
            # 提取真实未来轨迹: shape [1, 1, 64, 3] -> 取出最后一个时间步的位移
            gt_traj = data["ego_future_xyz"].cpu().numpy()
            
            # 最后一帧（未来6.4秒左右）的坐标，索引0是X轴(纵向偏移)，1是Y轴(横向偏移)
            final_x = gt_traj[0, 0, -1, 0]
            final_y = gt_traj[0, 0, -1, 1] 

            # 根据横向偏移动态打标签（阈值设为 2.5 米，约等于变道宽度的偏移）
            threshold = 2.5
            if final_y > threshold:
                scenario = "Turn Left / Lane Change Left"
            elif final_y < -threshold:
                scenario = "Turn Right / Lane Change Right"
            else:
                scenario = "Keep Lane / Straight"

            # 写入结果
            with open(args.output_csv, 'a', encoding='utf-8') as f:
                f.write(f"{clip_id},{final_x:.3f},{final_y:.3f},{scenario}\n")
                
            if i % 10 == 0 or i == 1:
                logger.info(f"[{i}/{len(clip_list)}] {clip_id} | Y轴偏移: {final_y:+.2f}m -> 标签: {scenario}")

        except Exception as e:
            logger.error(f"[{i}/{len(clip_list)}] 读取 {clip_id} 失败: {str(e)}")

    logger.info("✅ 标签生成完成！可以结合成绩单CSV进行Merge分析了。")

if __name__ == "__main__":
    main()
