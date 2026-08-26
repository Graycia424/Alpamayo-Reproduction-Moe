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
主要功能：实现增量轨迹与动作空间的分词器/编解码器 (Delta Trajectory Tokenizer)（即量化连续位置变量为离散词汇供 LLM 生成）。
数据流与模块关系说明：
1. 【输入来源】：正向量化时，接受来自物理环境记录或 `action_space/` 算出的连续位移相对值（Data Float Tensors）。
2. 【核心处理】：`DeltaTrajectoryTokenizer` 对这些物理值进行区间切分（Bins）实现量化（Quantize），变成离散词元（Tokens）；或实现基于概率分布的反量化计算。
3. 【输出去向】：编码结果（Tokens）流入语言模型 `alpamayo1_5.py` 的词表嵌入供训练；而反量化的结果解包输出给 `inference.py` 或仿真环境作为预测得到的执行动作（物理真值）。
"""

import einops
import numpy as np
import torch


class DeltaTrajectoryTokenizer:
    """增量轨迹Tokenizer (分词器/编码器)。
    不再直接将绝对轨迹送给模型，而是将其相对历史移动的变化量(Delta)进行量化为离散的Token(分词)以供常规LLM生成预测使用。
    """

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        load_weights: bool = False,
    ):
        """初始化轨迹分词器。

        参数：
            ego_xyz_min/max: 车辆每步可能位移(x, y, z)的极值物理区间（米为单位）。
            ego_yaw_min/max: 偏航角增量极值界限。
            num_bins: 将这一段连续模拟值分配成多少个离散的装箱（词汇量大小，默认1000格）。
            predict_yaw: 布尔值，设为True则不仅把坐标，也把朝向信息离散为Token并生成。
        """
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

    @property
    def vocab_size(self) -> int:
        """Token值被映射为整数集合 {0, 1, ..., vocab_size - 1} 中的数"""
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
        """核心编码函数：将连续的未来动作点序列加密量化为可以作为语言理解预测目标的离散Token组。

        注：该处理方式无关于绝对坐标系，而是对当前步与上一步坐标进行差分求增量，最后散列化。
        
        参数:
            hist_xyz/rot: （不使用，留作统一接口）
            fut_xyz (torch.Tensor): 连续未来的三维中心位置. 形状: (B, Tf, 3).
            fut_rot (torch.Tensor): 连续未来的车辆3D自转描述. 形状: (B, Tf, 3, 3).
            
        返回:
            torch.LongTensor: 量化编码完成后的索引标记Tokens. 形状: (B, num_tokens_per_trajectory).
        """
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        
        # 将原点（车辆位置全0）补充拼接至第一位以确保跟第一步之间也有增量概念
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1] # 求各个时刻相比上一时刻的delta差分增量
        
        # 定义极值域，将差异归一化至0~1范围
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        
        # 乘以词汇表数进行四舍五入即完成量化转为整型 ID，并且夹逼上下限防溢出
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        
        if not self._predict_yaw:
            # 如果不含yaw偏航，直出平移特征Token一维数组
            return einops.rearrange(xyz, "b n m -> b (n m)")
            
        # 如果需要将偏航角提取入Token:
        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])

        # 获取增量偏航的差分
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]

        # 利用sin和cos再atan2的手法规整偏航值域处于标准 [-pi, pi] 之间
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # 比例缩放并将偏航夹逼量化
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)

        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)  # 组合[x,y,z,yaw], 形状: (B, Tf, 4)
        return einops.rearrange(xyzw, "b n m -> b (n m)")

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """将模型生成的离散标记(Tokens) 解码并推演回其连续轨迹原始表示（三维坐标、三维旋转矩阵）。

        参数:
            hist_xyz / hist_rot (torch.Tensor): 同样只用作确定计算返回设备/类型，历史位置/转角。
            tokens (torch.LongTensor): 本批从模型推断采样的加密整数序列. 形状: (B, num_tokens_per_trajectory).

        返回:
            fut_xyz (torch.Tensor): 还原重组的高自由度浮点坐标矩阵 XYZ. 形如: (B, Tf, 3).
            fut_rot (torch.Tensor): 还原并复原的三维旋转位姿矩阵. 形状: (B, Tf, 3, 3).
        """
        del hist_tstamp
        m = 4 if self._predict_yaw else 3
        # 将展平的数组重塑为每时刻分离的样子(b, 步骤长T, 模态宽m)
        xyzw = einops.rearrange(tokens, "b (n m) -> b n m", m=m).to(hist_xyz.dtype)
        xyz = xyzw[..., :3]
        
        # 反归一化：比例 * 范围距 + 极小值
        xyz = xyz / (self.num_bins - 1)
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = xyz * (ego_xyz_max - ego_xyz_min) + ego_xyz_min
        # 因为编码的是delta微分，所以在解码时不断累加求原函数就能还原成整条连续轨迹
        fut_xyz = torch.cumsum(xyz, dim=1)
        
        # 没有直接推测偏航的情形下，采用多项式插值与位置导数近似推测自然状态下的朝向转角
        if not self._predict_yaw:
            xyz_cpu = fut_xyz.cpu().numpy().astype(float)
            fut_rot = get_yaw_rotation_matrices(xyz_cpu)
            fut_rot = torch.tensor(fut_rot, device=fut_xyz.device, dtype=fut_xyz.dtype)
            return fut_xyz, fut_rot, None
            
        # 否则手动恢复保存的Yaw变化量
        yaw_tokens = xyzw[..., 3]
        yaw = yaw_tokens.float() / (self.num_bins - 1)
        yaw = yaw * (self.ego_yaw_max - self.ego_yaw_min) + self.ego_yaw_min
        yaw = torch.cumsum(yaw, dim=1)

        # 把一维偏航角重建为遵循车偏航规则的 3x3 自转矩阵（沿Z轴发生转换）
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(cos_yaw)
        ones = torch.ones_like(cos_yaw)

        fut_rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        ).to(device=hist_rot.device, dtype=hist_rot.dtype)
        return fut_xyz, fut_rot, None


def get_yaw_rotation_matrices(trajectory, window_size=10, poly_order=3):
    """基于离散轨迹点 x(t) 和 y(t) 使用多项式位移求导曲线拟合方式以计算自车偏航(Yaw)旋转矩阵。

    当只知道坐标而不知道车头具体转到什么角度时，我们可以通过滑窗采样加微积分，近似估算出它“自然行走”时应保持的朝向切线斜率。

    参数:
        trajectory: 本批次的轨迹集点数组 np.array ，形如 (B, N, 3) 
        window_size: 多项式拟合局部滑雪窗口大小
        poly_order: 需要利用的多项式的阶数(曲线灵活度)

    返回:
        rotation_matrices: 所有位点的自转矩阵集，形状 (B, N, 3, 3)
    """
    B, N = trajectory.shape[:2]
    rotation_matrices = []

    for b in range(B):
        traj_batch = trajectory[b]  # (N, 3)
        batch_matrices = []
        batch_yaws = []

        for i in range(N):
            # Get window indices with padding for edges
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, start_idx + window_size)

            # Adjust window if at edges
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

            # Get points in window
            window_points = traj_batch[start_idx:end_idx]

            # Use time parameter t
            t = np.arange(len(window_points))

            # Fit polynomials to both x(t) and y(t)
            x_coeffs = np.polyfit(t, window_points[:, 0], poly_order)
            y_coeffs = np.polyfit(t, window_points[:, 1], poly_order)

            # Calculate derivatives at center point
            center_t = min(i - start_idx, window_size - 1)
            x_deriv = np.polyder(x_coeffs)
            y_deriv = np.polyder(y_coeffs)

            dx = np.polyval(x_deriv, center_t)
            dy = np.polyval(y_deriv, center_t)

            # 解析此时(t时刻)向量切线的切向方向，即为车辆此时的朝向(Yaw)角
            yaw = np.arctan2(dy, dx)
            batch_yaws.append(yaw)

            # 根据求得的 Yaw 值构建对应这帧时刻空间Z轴的偏航自转 3x3 矩阵
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            batch_matrices.append(rotation_matrix)

        rotation_matrices.append(batch_matrices)

    return np.array(rotation_matrices)
