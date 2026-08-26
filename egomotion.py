# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
本文件属于 Alpamayo 项目。
主要功能：专门用来记录和处理车辆自己怎么动的数据（比如车的位置、速度、加速度和转弯曲率）。

【与其他文件的依赖调用关系】
- 向上依赖（本文件调用了谁）：
  1. 调用了 `interpolation.py`：利用它来实现运动轨迹的平滑插值。
  2. 调用了 `tf.py`：利用里面的空间坐标系转换工具，让车身坐标能跟着世界坐标变化。
- 向下提供（谁调用了本文件）：
  1. 被 `dataset.py` 调用：当加载硬盘里的自动驾驶数据集时，由本文件负责把表格数据变成标准的“自车运动对象”。
  2. 被 `alpamayo1_5/action_space/unicycle_accel_curvature.py` 等底层算法调用：为他们提供基础的运动数据计算源。
"""
import dataclasses
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.transform as spt

import interpolation, tf


@dataclasses.dataclass
class EgomotionState(interpolation.Interpolatable, tf.Transformable):
    pose: spt.RigidTransform = dataclasses.field(
        metadata=interpolation.Interpolatable.DEFAULT_RIGID_TRANSFORM_INTERPOLATION
        | tf.Transformable.POSE
    )
    velocity: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR | tf.Transformable.VECTOR
    )
    acceleration: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR | tf.Transformable.VECTOR
    )
    curvature: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR
    )

    @classmethod
    def from_egomotion_df(cls, egomotion_df: pd.DataFrame) -> Self:
        return cls(
            pose=spt.RigidTransform.from_components(
                rotation=spt.Rotation.from_quat(egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy()),
                translation=egomotion_df[["x", "y", "z"]].to_numpy(),
            ),
            velocity=egomotion_df[["vx", "vy", "vz"]].to_numpy(),
            acceleration=egomotion_df[["ax", "ay", "az"]].to_numpy(),
            curvature=egomotion_df[["curvature"]].to_numpy(),
        )
