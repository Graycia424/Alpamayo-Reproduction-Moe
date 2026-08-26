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
主要功能：实现将连续控制空间转换为分类索引/离散槽位的离散动作空间接口机制 (`DiscreteTrajectoryTokenizer` / `DiscreteActionSpace`)。
数据流与模块关系说明：
1. 【输入来源】：依赖并包装由 `action_space.py` 初始化后的真实连续物理控制序列 (比如车辆速度，曲率等)。这些连续数据从基础数据集抽取出。
2. 【核心处理】：进行分桶算法（Bins）的运算，连续值在 `dims_min` 与 `dims_max` 的尺度内被强制划分为多个离散标号档位；同时也包含解码对应还原的过程。
3. 【输出去向】：产生的整数类别的 Token 标志能够无缝对接给 `delta_tokenizer.py` 并作为 `alpamayo1_5.py` LLM 大语言模型的分类预测头类别数 (例如 256 甚至上万个 Vocab Size 的分类预测)。
"""

from typing import Any

import hydra.utils as hyu
import torch

from alpamayo1_5.action_space.action_space import ActionSpace


class DiscreteTrajectoryTokenizer:
    """Discrete trajectory tokenizer."""

    def __init__(
        self,
        action_space_cfg: dict[str, Any],
        dims_min: list[float],
        dims_max: list[float],
        num_bins: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the tokenizer."""
        self.action_space: ActionSpace = hyu.instantiate(action_space_cfg)
        assert len(dims_min) == len(dims_max) == self.action_space.get_action_space_dims()[-1]
        self.dims_min = dims_min
        self.dims_max = dims_max
        self.num_bins = num_bins

    @property
    def vocab_size(self) -> int:
        """Tokens are integers from the set {0, 1, ..., vocab_size - 1}"""
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Encodes the trajectories as discrete tokens.

        We assume the the future tstamp is consistent with the future trajectory.

        Args:
            hist_xyz: The history xyz coordinates.
            hist_rot: The history rotation matrices.
            fut_xyz: The future xyz coordinates.
            fut_rot: The future rotation matrices.
            hist_tstamp: The history timestamps.
            fut_tstamp: The future timestamps.

        Returns:
            tokens: The encoded tokens. Shape: (B, num_tokens_per_trajectory).
        """
        batch_size = fut_xyz.shape[0]
        action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = (action - dims_min) / (dims_max - dims_min)
        action = (action * (self.num_bins - 1)).round().long()
        action = action.clamp(0, self.num_bins - 1)
        return action.reshape(batch_size, -1)

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Decodes the given tokens into future trajectories.

        We assume the the future tstamp is consistent with the future trajectory.

        Args:
            hist_xyz: The history xyz coordinates.
            hist_rot: The history rotation matrices.
            tokens: The tokens to decode.
            hist_tstamp: The history timestamps.

        Returns:
            fut_xyz: The decoded future xyz coordinates.
            fut_rot: The decoded future rotation matrices.
            None: The future timestamps are not decoded.
        """
        action = tokens.reshape(-1, *self.action_space.get_action_space_dims()).to(hist_xyz.dtype)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = action / (self.num_bins - 1)
        action = action * (dims_max - dims_min) + dims_min
        fut_xyz, fut_rot = self.action_space.action_to_traj(action, hist_xyz, hist_rot)
        return fut_xyz, fut_rot, None
