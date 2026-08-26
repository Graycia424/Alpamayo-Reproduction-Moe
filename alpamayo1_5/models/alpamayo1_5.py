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
主要功能：实现基于 Transformer 架构的 Alpamayo 1.5 视觉-语言-动作 (VLA) 的核心模型类。
数据流说明：
1. 模型接收处理后的传感器输入 (Video/Camera帧) 与控制 Prompt。
2. 数据流经 Embedding 层并合并，输入至多模态自回归推断核心（LLM backbone + Vision Encoder）。
3. 前向传播中不仅生成链式思维(Chain of Thought)文本令牌(Token)，同时也并行输出自车下一步基于环境反馈的连续动作信号 (预测轨迹与位姿)。
"""

import copy
from functools import partial
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from alpamayo1_5.action_space import ActionSpace
from alpamayo1_5.models.base_model import ReasoningVLA
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.diffusion.base import BaseDiffusion
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_5.nav_utils import remove_nav_text

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """
    专家网络的Logits处理器：在生成的分布里屏蔽掉（mask）离散轨迹Token的概率。
    这确保语言模型不会误输出用于预测离散轨迹的专属Token，提升思维链（CoT）生成的表现。
    """

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """初始化函数。

        参数:
            traj_token_offset: 用于表示轨迹标记(tokens)开始的索引偏移量。
            traj_vocab_size: 轨迹标记所占用的词汇表大小。
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """调用函数，将离散的控制/动作 Token 分数掩除。

        由于专家网络直接产生连续/混合形式的动作分布，不需要产生传统离散化轨迹标记，通过屏蔽让文本生成更加纯净。

        参数:
            input_ids: 已经生成的Token ID序列。
            scores: 当前步模型输出的词汇表预测分数分布。

        返回:
            torch.FloatTensor: 将代表轨迹的Token预测分数设为负无穷大 (-inf) 后的分数张量。
        """
        # 直接把轨迹相关的Token得分位赋为了负无穷
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


class Alpamayo1_5(ReasoningVLA):
    """Alpamayo 1.5版，用于推理的视觉-语言-动作 (Reasoning VLM/VLA) 专家模型。"""

    config_class: type[Alpamayo1_5Config] = Alpamayo1_5Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: Alpamayo1_5Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        # 继承和初始化基础的多模态语言模块
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # 这里的"专家(expert)"特指在通用模型之后外挂处理连续控制信号的子网络
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # 根据设计，专家子模型共享上一层的词嵌嵌入(embed_tokens)，因此删除其本身的以节省显存
        del self.expert.embed_tokens

        # 初始化动作空间（例如将车辆运动分解为加速度和曲率）
        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        # 初始化扩散模型（Diffusion），作为底层的生成器去预测未来的物理动作分量
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        # 进/出映射投影网络(Projection)
        # action_in_proj将输入的动作历史表征转换为大语言模型的隐藏层特征规模
        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        # action_out_proj将从大语言模型得出的特征转换为后续扩散网络所需的参数维数
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # 类型对齐：将动作生成的部分转化成与专家(LLM部分)相同的精度格式以确保端到端不出冲突
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    @staticmethod
    def _find_eos_offset(
        sequences: torch.Tensor,
        eos_token_id: int,
        device: torch.device,
        warn: bool = True,
    ) -> torch.Tensor:
        """寻找每个序列中第一个eos_token_id（结束符标记）的位置，并返回 偏移量 = 位置 + 1。

        当找不到eos_token_id时，回退至截取最后一个token的位置。
        这个返回的偏移量用来标记 VLM(视觉语言大模型)的自回归文本生成 token与专家级扩散网络(diffusion) 附加token之间的天然边界。
        """
        b_star = sequences.shape[0]
        mask = sequences == eos_token_id
        has_eos = mask.any(dim=1)  # [b_star] 找是否存在EOS符
        if warn:
            for i in range(b_star):
                if not has_eos[i]:
                    logger.warning(
                        f"在生成的序列中未找到 <traj_future_start> 标记位。发生于序列号 {i}"
                    )
        eos_positions = mask.int().argmax(dim=1)  # [b_star], 找到第一次出现的下标
        last_positions = torch.full((b_star,), sequences.shape[1] - 1, device=device)
        return torch.where(has_eos, eos_positions, last_positions) + 1

    @staticmethod
    def _build_expert_pos_ids_and_attn_mask(
        offset: torch.Tensor,
        rope_deltas: torch.Tensor,
        kv_cache_seq_len: int,
        n_diffusion_tokens: int,
        b_star: int,
        device: torch.device,
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构建用于专家降噪器（Expert Denoiser）的位置编码ID以及4D注意力掩码(Attention Mask)。

        参数:
            offset: [b_star] —紧接着 <traj_future_start> 之后的token位置。
            rope_deltas: [b_star, 1] — 来自VLM输出累积的旋转位置编码(RoPE)偏移参数。
            kv_cache_seq_len: 在KV Cache中已经存储的上下文序列长度。
            n_diffusion_tokens: 我们需要在后边追加给专家网络（扩散网络）处理的token总数。
            b_star: 批次大小 (B * num_return_sequences)。
            device: torch计算设备(cuda/cpu)。
            prefix_mask: [b_star, L] 可选的一维注意力掩码；其中0将意味着要在此对KV Cache掩掉注意力。

        返回:
            position_ids: [3, b_star, n_diffusion_tokens] — 适用于Qwen2.5-VL底座模型的三维 RoPE 标识。
            attention_mask: [b_star, 1, n_diffusion_tokens, KV] — 浮点型的四维注意力掩码张量
                (0 = 允许建立注意力关联, -inf = 被遮蔽掉不可见)。
        """
        # Qwen2.5-VL 运用了三要素 (时间temporal, 高度height, 宽度width) 复合的 RoPE(旋转位置编码)
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

        # 尺寸为 [b_star, H, Q, KV] — 我们会将从偏移点到生成扩散网络token之间的空白段进行掩膜屏蔽
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        # 向前传播输入的左侧填充掩盖（left-padding mask）到KV前缀区
        if prefix_mask is not None:
            # [b_star, H, Q, KV]
            input_mask = prefix_mask[:, None, None, :]
            attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
                input_mask == 0,
                torch.finfo(attention_mask.dtype).min,
                attention_mask[:, :, :, : input_mask.shape[-1]],
            )

        return position_ids, attention_mask

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过自回归VLM模型的展开，从多模态输入数据中生成并抽样出未来的行车轨迹。

        参数:
            data: 输入数据，通常包含 "tokenized_data"（图像+文本的嵌入），
                  以及 "ego_history_xyz", "ego_history_rot" 等自我运动历史数据。
            top_p: 核采样（Nucleus Sampling）的概率阈值。
            top_k: 截断采样的K值大小。
            temperature: 采样的温度系数，控制生成随机性。
            num_traj_samples: 每次轨迹采样的样本数量。
            num_traj_sets: 独立进行采样的集合次数。
            diffusion_kwargs: 扩散模型采样的额外参数和配置。
            *args: 变长位置参数。
            **kwargs: 变长关键字参数。

        返回:
            pred_xyz: 预测结果中的未来中心点平移量(xyz坐标张量)。
            pred_rot: 预测结果中的未来车辆自旋量(旋转矩阵形式)。
            extra: (根据额外参数选择性返回) 包含文本序列生成时的中间结果。
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "在推理阶段时，只支持一种轨迹组。"
        
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        # 将历史连续数据通过token转换技术融合进Token空间
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) 第1阶段：运行视觉语言大模型(VLM)的自回归文本生成以产生自然语言思考过程
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # 使用自定义的停止准则，在产生结束符(EOS, <traj_future_start>) 及其后携带一个额外token后立马停止，
        # 从而能在新出的token后面随时更新KV Cache以接入专家系统
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        # 在生成过程中遮蔽特定离散标记
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        
        # 启动主模型预热与推理
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # 手动将被阶段切断的EOS后的垃圾pad重新替换处理以保障后续正确截断
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values # 大提取出目前前缀Prompt中缓存的信息(VLM运算成果)
        prefill_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = self._find_eos_offset(
            sequences=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            device=device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
            
        # 根据我们切断出来的时间偏移量，生成接下来供给后处理网络的坐标位次和多维遮罩
        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False # 非因果机制指模型能够看到时间线上之后的序列(双向注意)

        # 2) 第二阶段：定义一个去噪步骤闭包(step_fn)，能够接收充满噪声的动作以及扩散时间步(timestep)
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x形状: (B*, *action_dim)
            # t形状能被广播到 x的前几个维度
            b_star = x.shape[0]
            # 把充满噪声的前向物理动作投影编码为接下来的n个专家隐藏特征标记层词向量
            # 期望出的维度(Expect shape): (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # 在上面保存好的VLM prompt文本特征缓存序列(KV Cache)上，运行仅基于未来标记专家的外挂网络
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # 在单次降噪后裁剪并剥去多余刚刚生成步的缓存，保证每次降噪步骤都以相同的历史长度作为起点
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # 获得最后一层特征 (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            
            # 使用输出投射器将它翻译成还原预测噪声或下一次向量的连续信息特征
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) 第三阶段：通过给每个输入派发多个样本并在动作空间上通过刚刚定义的函数进行扩散(Diffusion)反向采样
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # 重复扩散前的历史基线坐标，使其数量和扩散结果的多个批次相互对齐
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        # 取出采样的动作序列并根据之前的点和方向，经过曲率/动力学公式解析还原为未来预测世界物理轨迹
        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) 重塑并变换成规定的返回格式维数 (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # 选择性提取VLM伴随此推断所附带产生的任何原始思维链长本文回复标记
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # 重排本文将其与结果一致
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    @torch.no_grad()
    def sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """带CFG (Classifier-Free Guidance, 无分类器引导)与导航指令处理的扩展轨迹推断采样器。
        
        此方法除了执行基础的多模态推断外，同时还会构建一个没有自然语言导航提示(“无指导”/unguided)的降维缓存分支。
        在执行物理动作降噪时运用CFG技术，放大有导航指令情况下的动作网络响应倾向，使得遵守指令的概率增高。

        参数:
            data: 输入数据字典（含图像、指令、车辆运动数据）。
            top_p: 核采样（Nucleus Sampling）的保留概率截断值。
            top_k: Top-K 采样的阈值设置。
            temperature: VLM端语言生成的温度指数，控制文本随机性（发散度）。
            num_traj_samples: 单次推理批中生成的预测物理轨迹的个体数量。
            num_traj_sets: 并行的独立采样组的倍数数量。
            diffusion_kwargs: 直接透传给底层扩散（Diffusion）模型的其余采样超参数。
            *args: 变长列表格式的位置参数。
            **kwargs: 变长的可选键值对参数字典。

        返回:
            pred_xyz: 推断的动作中，预测的未来位置平移张量 (B, num_traj_sets, num_traj_samples, ... )。
            pred_rot: 推断的动作中，预测的未来姿态旋转张量 (与上一参数同维)。
            extra: (根据额外参数选择性返回) 生成这组参数时，VLM模型内心思考经过(Token)。
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) 第1阶段: 启动视觉语言模型(VLM)的递归推导自回归文本生成阶段
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # 设立断点准则：在抓到轨迹起点指令标识和附带的预判Token之后踩刹车
        # 注意:之所以要求额外多生成一个词才停止，是为了在加入专家网络算子的时候能把其先垫入KV缓存更新里
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        # 在构造无向/未制导引导对照组缓存之前，释放不需要的大型生成Logit中间阵列
        del vlm_outputs.logits
        torch.cuda.empty_cache()
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # 人工消除截后产生的一连串补丁(Padding)字符残余
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = self._find_eos_offset(
            sequences=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            device=device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        """
        # (后续处理与计算：针对无监督提示截断面与专家遮罩计算等)
        """
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prompt_cache.get_seq_length(),
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        # 2) construct unguided kv cache
        # Build unguided input_ids by removing <|route_start|>...<|route_end|> span
        unguided_input_ids = []
        for i in range(input_ids.shape[0]):
            unguided_input_ids.append(remove_nav_text(input_ids, self.tokenizer, i)[0])
        unguided_input_ids = torch.nn.utils.rnn.pad_sequence(
            unguided_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        unguided_prefix_mask = unguided_input_ids.ne(self.tokenizer.pad_token_id).long()

        # Step 1: Prefill unguided prefix ONCE with original batch (B samples).
        # Vision encoder runs only once — no pixel_values repetition needed.
        unguided_prefill_outputs = self.vlm(
            input_ids=unguided_input_ids,
            attention_mask=unguided_prefix_mask,
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            pixel_values=tokenized_data.get("pixel_values"),
            use_cache=True,
            logits_to_keep=1,
        )

        # Step 2: Repeat KV cache for n_samples_total (cheap memory copy, no recomputation)
        # Free the prefill outputs first — we only need the KV cache, not the logits
        unguided_prompt_cache = unguided_prefill_outputs.past_key_values
        del unguided_prefill_outputs
        torch.cuda.empty_cache()
        unguided_prompt_cache.batch_repeat_interleave(n_samples_total)

        # Step 3: Forward generated_tokens (which differ per sample) using the repeated
        # KV cache. No pixel_values needed — images are already encoded in the cache.
        generated_tokens = vlm_outputs.sequences[:, input_ids.shape[1] :]
        unguided_prefix_len = unguided_input_ids.shape[1]
        gen_len = generated_tokens.shape[1]

        prefix_mask_repeated = unguided_prefix_mask.repeat_interleave(n_samples_total, dim=0)
        gen_mask = generated_tokens.ne(self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([prefix_mask_repeated, gen_mask], dim=1)

        cache_position = torch.arange(
            unguided_prefix_len,
            unguided_prefix_len + gen_len,
            device=device,
            dtype=torch.long,
        )

        unguided_vlm_outputs = self.vlm(
            input_ids=generated_tokens,
            attention_mask=full_attention_mask,
            past_key_values=unguided_prompt_cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        unguided_prompt_cache = unguided_vlm_outputs.past_key_values
        del unguided_vlm_outputs.logits
        torch.cuda.empty_cache()

        full_unguided_tokens = torch.cat(
            [torch.repeat_interleave(unguided_input_ids, n_samples_total, dim=0), generated_tokens],
            dim=1,
        )
        unguided_offset = self._find_eos_offset(
            sequences=full_unguided_tokens,
            eos_token_id=eos_token_id,
            device=device,
            warn=False,
        )
        unguided_prefix_mask_repeated = torch.repeat_interleave(
            unguided_prefix_mask, n_samples_total, dim=0
        )
        unguided_position_ids, unguided_attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=unguided_offset,
            rope_deltas=unguided_vlm_outputs.rope_deltas,
            kv_cache_seq_len=unguided_prompt_cache.get_seq_length(),
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=unguided_prefix_mask_repeated,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 3) 第3阶段：定义用于降噪的高级步进函数，它同时需要处理含有注意力矩阵在内的多模态缓存
        # 相比普通的 rollout，这里的 step_fn 被闭包扩充了接受KV Cache等VLM历史的能力
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            prefill_seq_len = past_key_values.get_seq_length()
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            past_key_values.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 4) 第4阶段：真正在动作控制空间中应用带CFG引导和非引导交替对照的多样本扩散求解器
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=partial(
                step_fn,
                past_key_values=prompt_cache,            # 正向指引导航文本缓存
                attention_mask=attention_mask,
                position_ids=position_ids,
            ),
            unguided_step_fn=partial(
                step_fn,
                past_key_values=unguided_prompt_cache,   # 被强迫截断删除导航信息的无控缓存
                attention_mask=unguided_attention_mask,
                position_ids=unguided_position_ids,
            ),
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # 把历史坐标数据扩展复制以跟扩散出的多份动作采样尺寸对齐
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        # 将生成的动作量场(比如离散偏航角角速度数组) 变换/积分解码为空间中的实数规划目标轨迹
        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 5) 将序列整理重塑为期望的三维结果张量结构 (Batch, num_traj_samples, 长度,特征维)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # 可选特性：从原本的生成中把文本语言序列单独按字符串解包出来一并反回给前端页面展示
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo1_5", Alpamayo1_5Config)
AutoModel.register(Alpamayo1_5Config, Alpamayo1_5)
