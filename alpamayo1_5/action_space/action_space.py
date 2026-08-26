# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
本文件属于 Alpamayo 项目。
主要功能：定义通用“动作空间 (Action Space)”的抽象基类，规范车辆/机器人在物理空间的动作解析和重构接口。
数据流与模块关系说明：
1. 【输入来源】：定义了接受从底层的运动学物理状态或高层的模型模型推断的动作输出作为输入的协议规范。
2. 【核心处理】：确立了核心的两大抽象方法 `traj_to_action` (从轨迹算动作) 和 `action_to_traj` (从动作生轨迹) 的骨架结构协议。
3. 【输出去向】：作为基类模块被 `discrete_action_space.py`、`unicycle_accel_curvature.py` 继承实现出专门应对离散空间或独轮车模型的动作逻辑层，为 `inference.py` 模型采样评估与数据提取服务。
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ActionSpace(ABC, nn.Module):
    """Action space base class for the trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform the future trajectory to the action space.

        Args:
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            action: (..., *action_space_dims)
        """

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space to the trajectory.

        Args:
            action: (..., *action_space_dims)
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
        """

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]:
        """Get the dimensions of the action space.

        Returns:
            action_space_dims: the action space dimensions
        """

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if the action is within the bounds.

        By default, we assume the action is within bounds (dummy implementation).

        Args:
            action: (..., *action_space_dims)

        Returns:
            is_within_bounds: (...,)
        """
        num_action_dims = len(self.get_action_space_dims())
        batch_shape = action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)
