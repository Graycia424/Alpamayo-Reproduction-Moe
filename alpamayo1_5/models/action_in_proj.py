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
主要功能：定义“动作输入投影 (Action Input Projection)”相关的神经网络结构模块，实现数值形式的动作向模型嵌入空间的投影。
数据流与模块关系说明：
1. 【输入来源】：接受由 `action_space/` 相关文件提供或解析处理后的，形如历史连续动作序列、行驶轨迹坐标等原始物理连续张量。
2. 【核心处理】：作为可学习的特征映射头（Encoder/Projector），通过多层感知机(MLP)或线性映射附加层高归一化(RMSNorm)结构，将低维的连续物理动作数据投射成高维嵌入表示 (Embeddings)。
3. 【输出去向】：转换后的高维张量输出至 `alpamayo1_5.py`（或主模型框架），与其他模态数据（文本、视频Token）在相同的维度上无缝拼接，共同送入自回归Transformer的计算图中。
"""

import math

import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    """均方根层归一化 (Root Mean Square Normalization) 的实现。
    比传统LayerNorm计算开销更小且同样有效，是目前大模型常用的归一化手段。
    """
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """对输入张量沿最小维度执行除以均方根的归一化。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播处理归一化和可学习放缩。"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLPEncoder(nn.Module):
    """基础的多层感知机 (MLP) 编码器。
    用于将特定维度的特征逐步投射(Project)到我们想要的目标维度以对齐大模型隐层。
    """

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} 必须大于或等于 1"

        # 第一层：线性变换加SiLU激活函数
        enc_layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        # 叠加剩余的隐藏层
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                # 最后一层：归一化之后只做线性映射，不用激活函数处理输出
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )

        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: (B, C) -> (B, outdim)"""
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """采用对数间隔频率来强化数值特征表征的改良版傅里叶特征编码器(Fourier Feature Encoder)。
    (常用于位置编码或将连续数值散列化以帮助高频信息被网络捕获)
    """

    def __init__(self, dim: int, max_freq: float = 100.0):
        """初始化傅里叶编码器 V2版。

        参数:
            dim: 编码器输出维度。必须是偶数，因为它将被平分为正弦(sin)和余弦(cos)部分。
            max_freq: 用于创建对数级频率的最大截至频率，默认为 100.0。
        """
        super().__init__()
        half = dim // 2
        # 在对数图上生成频率点阵
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :], persistent=False)  # 将频率张量保存入显存但不随模型存入权重表 (1, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """傅里叶编码器的前向传播。

        参数:
            x: 任意形状的张量输入 (..., )。

        返回:
            拥有张量形状(..., dim)的傅里叶编码后特征。
        """
        arg = x[..., None] * self.freqs * 2 * torch.pi  # 得到正余弦的自变量 (*, half_dim)
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(torch.nn.Module):
    """改良的面向逐个路点的动作特征输入映射模块，主要用于对接扩散模型(Diffusion)的条件调节环节。

    它依赖携带独立对数频率的 FourierEncoderV2 处理信息，包含层归一化模块。
    核心作用：将带有扩散时间步长(Timestep)的连续自动驾驶动作序列提升/投射到大模型要求的高维隐层表示。
    """

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        """初始化面向路点的动作投影网络。

        参数:
            in_dims: 输入维度的清单。通常只有最后一个元素指明了要分离出多少个需要独立作傅立叶编码的底层动作维度数（如平移控制、旋转度）。
            out_dim: 映射出的目标尺寸(比如对齐语言模型通道维度数 4096或8192)。
            num_enc_layers: 用于映射的感知机层数，默认 4层。
            hidden_size: 内部隐藏层的尺寸规模，默认 1024。
            max_freq: 傅里叶特征表征上限频率阈值，默认 100.0。
            num_fourier_feats: 每个基础数值要分解出多少个傅里叶切片，默认 20 个切片。
        """
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        # 创建给输入动作(纵向、横向、转向等连续基准特征)各自对应的傅里叶频分器
        sinus = []
        for _ in range(in_dims[-1]):
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq))
        self.sinus = nn.ModuleList(sinus)
        # 为扩散步长时间点(timestep)准备一个独立编码器，以告知模型它处理得是加噪到什么程度的特征
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        
        # 统计组合好的所有频分维度数
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        # 送入主多层感知机将宽阔复杂的特征重新提取打包为 `out_dim` 尺寸
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)  # 提供标准层归一化防止内部协方差偏移

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """投影层处理：传入路点与对应扩散时刻生成最终高维隐向量。

        参数:
            x: 原动作张量，形状通常为 (batch_size, num_waypoints预设路点数目, action_dim物理维度).
            timesteps: 当前扩散步骤对应的时间索引，形状为 (batch_size, ...)，
                最后一个维度用来推断步骤频率。

        返回:
            正则化并投影扩维后的动作表示信号，最终输出形状：
            (batch_size批次大小, num_waypoints路径点数量, out_dim映射维度)。
        """
        B, T, _ = x.shape

        # 对传入的每个动作维展开并独立利用其专属的傅立叶分解器取得高频切片，最后横向拼接起来
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        
        # 将一维离散的时间步同样转换成高级数值分布表征特征
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        # 时间步对于当下的所有预测路点在这一时刻显然都是共用的，所以复印拓展长度与路径对齐 (T份)
        timestep_feats = timestep_feats.repeat(1, T, 1)
        
        # 物理控制特征 + 时间步长调节特征进行拼接并展薄成2维丢入网络求积，最终还原折叠形态
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
