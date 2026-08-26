# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
本文件属于 Alpamayo 项目。
主要功能：作为整个项目的“数据源底座”，提供自动驾驶数据集的下载与访问接口。

【与其他文件的依赖调用关系】
- 向上依赖（本文件调用了谁）：
  1. 调用了 `video.py`：利用它把读取到的字节流压缩视频还原解析成图像帧。
  2. 调用了 `egomotion.py`：利用它将读取的物理轨迹数据转换为自车运动（EgomotionState）对象。
- 向下提供（谁调用了本文件）：
  1. 被 `alpamayo1_5/load_physical_aiavdataset.py` 调用：为其提供具体的视频和轨迹片段，用于拼装成喂给大模型的 Batch 输入。
  2. 被 `inference.py` 和 `inference_evaluation.py` 调用：在端到端推理和评估时，必须先实例化本文件中的 `PhysicalAIAVDatasetInterface` 作为全家桶数据入口。
"""
import dataclasses
import json
import io
import logging
import pathlib
import types
import zipfile
from typing import Any, Iterable
import os
import pandas as pd
import egomotion, video



logger = logging.getLogger(__name__)


class PhysicalAIAVDatasetInterface():
    """Hugging Face上PhysicalAI-Autonomous-Vehicles数据集的交互接口。

    该类主要用于访问和加载自动驾驶数据集，包括车辆姿态(egomotion)和多视角摄像头的视频数据。

    Attributes/属性:
        revision (`str`): Git版本号，目前可忽略。
        token (`str | bool | None`): 访问令牌。
        cache_dir (`str | pathlib.Path | None`): 缓存目录路径。
        local_dir (`str | pathlib.Path | None`): 本地下载目录路径。
        features (`Features`): 数据集特征配置类，通过`.`操作符自动补全特征名。
        clip_index (`pd.DataFrame`): 映射视频片段`clip_id`到由于存储分块所对应的`chunk`索引表。
        sensor_presence (`pd.DataFrame`): 记录每段`clip_id`可用的传感器列表（包括雷达、相机等）。
        chunk_sensor_presence (`pd.DataFrame`): 按存储数据块(chunk)聚合的可用传感器记录。
    """

    def __init__(
        self,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        super().__init__(
            # repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
            # repo_type="dataset",
            # revision=revision,
            # token=token,
            # cache_dir=cache_dir,
            # local_dir=local_dir,
            # confirm_download_threshold_gb=confirm_download_threshold_gb,
        )
        features_df = pd.read_csv("features.csv", index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        # 读取每个视频剪辑对应的文件块(Chunk)索引
        self.clip_index = pd.read_parquet("clip_index.parquet")
        # 读取每个剪辑拥有的传感器硬件配置信息
        self.sensor_presence = pd.read_parquet("/data/PhysicalAI-Autonomous-Vehicles/metadata/sensor_presence.parquet")
        # 将片段传感器分布按照存储块(chunk)聚集，方便下载
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    def download_metadata(self) -> None:
        """下载数据集的元数据记录(metadata)，例如用于选择剪辑及数据分块（chunk）的数据信息表。"""
        self.metadata = {
            pathlib.Path(f).stem: pd.read_parquet(f) for f in self.download_repo_tree("metadata/")
        }

    def get_clip_chunk(self, clip_id: str) -> int:
        """通过视频片段短ID获取该片段所在的文件块(chunk)编号。"""
        return self.clip_index.at[clip_id, "chunk"]

    def get_clip_feature(self, clip_id: str, feature: str, types: str, allowed_camera_features: list = None) -> Any:
        """
        通过传入`clip_id`、特征名和类型，读取并返回数据集内具体的一个多模态特征实例。

        参数：
            clip_id (str): 测试片段的唯一ID标识。
            feature (str): 诸如相机视角名(e.g., 'camera_front_wide_120fov')。
            types (str): 特征载体类型，比如 "egomotion" 或 "camera"。
            allowed_camera_features (list): 允许加载的相机特征名列表（如只加载1/2/4摄像头）。
        """
        # 数据集在本地存放的基准路径
        base_url = "/path/to/data/PhysicalAI-Autonomous-Vehicles"

        # 如果类型是自身运动(自车底盘运动学特征)
        if types == "egomotion":
            egomotion_path = os.path.join(
                base_url,
                "labels",
                "egomotion",
                f"{clip_id}.egomotion.parquet"
            )

            print("Loading egomotion from:", egomotion_path)

            if not os.path.exists(egomotion_path):
                raise FileNotFoundError(f"Egomotion file not found: {egomotion_path}")

            # 读取包含自车移动位姿和四元数的parquet文件并产生插值采样器(Interpolator)，因为帧率不同
            egomotion_df = pd.read_parquet(egomotion_path)

            return egomotion.EgomotionState.from_egomotion_df(
                egomotion_df
            ).create_interpolator(
                egomotion_df["timestamp"].to_numpy() # 支持按指定时间戳插值读取底盘信息
            )

        # 如果提取特征类别是相机原始视频流
        if types == "camera":
            if allowed_camera_features is not None and feature not in allowed_camera_features:
                raise ValueError(f"Camera feature {feature} is not in allowed_camera_features: {allowed_camera_features}")
            # 找到对应相机的 mp4 压缩视频及其对于每帧时刻精准的时间戳文件
            video_path = os.path.join(
                base_url,
                "camera",
                feature,
                f"{clip_id}.{feature}.mp4"
            )

            timestamps_path = os.path.join(
                base_url,
                "camera",
                feature,
                f"{clip_id}.{feature}.timestamps.parquet"
            )

            print("Loading video from:", video_path)

            # 导出视频的精准时间戳（在后续与egomotion及指令结合的关键）
            timestamps = pd.read_parquet(timestamps_path)["timestamp"].to_numpy()

            with open(video_path, "rb") as f:
                video_bytes = f.read()

            video_data = io.BytesIO(video_bytes)

            # 返回带有内建时间戳解压器的视频按需阅读器，可以根据传入的期望时间解码出图像帧
            return video.SeekVideoReader(video_data, timestamps)
            


class Features:
    """定义表现数据集存储特征和它们打包在HuggingFace上格式的抽象类。"""

    def __init__(self, features_df: pd.DataFrame) -> None:
        self.features_df = features_df

        # 动态创建便于通过点操作符读取属性的字典（如 self.CAMERA.CAMERA_FRONT_WIDE_120FOV ）
        # 以及为某个传感器大类创建 `.ALL` 属性打包全部集合，方便业务方解包。
        self.ALL = set()
        for directory, directory_features in self.features_df.groupby("directory"):
            setattr(
                self,
                directory.upper(),
                types.SimpleNamespace(
                    **{feature.upper(): feature for feature in directory_features.index},
                    ALL=set(directory_features.index),
                ),
            )
            self.ALL.update(getattr(self, directory.upper()).ALL)

    def get_chunk_feature_filename(self, chunk_id: int, feature: str):
        """Returns the chunk feature filename within the dataset repo."""
        return self.features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> list[str]:
        """Returns the files within a chunk feature zip corresponding to `clip_id`."""
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"{feature=} is not chunked as zip files.")
        return {k: v.format(clip_id=clip_id) for k, v in templates.items()}