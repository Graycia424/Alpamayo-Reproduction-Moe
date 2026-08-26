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
主要功能：定义 Alpamayo 1.5 系列架构独有的配置类。
数据流与模块关系说明：
1. 【输入来源】：该文件包含硬编码或外部加载进来的超参数设置。
2. 【核心处理】：继承 `ReasoningVLAConfig` (`models/base_model.py`)，重写/注册特定模型类别表示或针对版本特化的参数（比如 `model_type="alpamayo1_5"`）。
3. 【输出去向】：配置对象会被 `alpamayo1_5.py` 的加载工具用来正确分配网络算子参数（如隐藏层大小，Tokenizer 设置等）。
"""

from typing import Any

from alpamayo1_5.models.base_model import ReasoningVLAConfig


class Alpamayo1_5Config(ReasoningVLAConfig):
    """
    Alpamayo 1.5 发布模型的参数配置类。
    继承自底层视觉-语言-动作 (VLA) 的推型基类配置。
    """

    model_type = "alpamayo1_5" # 模型类型标识

    def __init__(
        self,
        diffusion_cfg: dict[str, Any] | None = None,      # 扩散模型（用于动作生成）的配置字典
        action_space_cfg: dict[str, Any] | None = None,   # 动作空间（如加速度、曲率等控制量离散化）的配置
        action_in_proj_cfg: dict[str, Any] | None = None, # 历史动作映射入模型内部表示（Input Projection）的配置
        action_out_proj_cfg: dict[str, Any] | None = None,# 模型内部表示映射出具体连续/离散动作（Output Projection）的配置
        expert_cfg: dict[str, Any] | None = None,         # 混合专家网络（MoE）结构中特定专家模块配置
        keep_same_dtype: bool = True,                     # 是否维持内部权重在相同的数据类型（如 bfloat16）
        expert_non_causal_attention: bool = True,         # 专家网络是否采用非因果注意力机制（即可以看到前后上下文）
        include_camera_ids: bool = False,                 # 输入条件中是否显式合并相机视角ID标识符
        include_frame_nums: bool = False,                 # 输入条件中是否显式合并相机的帧时间序号标识符
        **kwargs: Any,                                    # 其他基础LLM或VLM传往父类的默认参数
    ) -> None:
        super().__init__(**kwargs)
        # 将传入参数持久化为类的属性
        self.diffusion_cfg = diffusion_cfg
        self.action_space_cfg = action_space_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_cfg = expert_cfg
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums
