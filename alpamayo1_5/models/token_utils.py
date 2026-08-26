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
主要功能：动作和轨迹 Token 的解析、停止及逻辑惩罚约束工具（Token Utilities）。
数据流与模块关系说明：
1. 【输入来源】：对接来自大语言模型即 `models/alpamayo1_5.py` 的自回归输出概率（Logits）或者刚刚生成的 Token 序列流。
2. 【核心处理】：拦截并在 Logits 层级约束不合法动作的概率(`TokenConstraintLogitsProcessor`)；监听生成流以匹配轨迹起始、结束符并触发停止策略（`CustomStoppingCriteria`）。
3. 【输出去向】：约束后的纯净动作 Token 序列会被交付给 `delta_tokenizer.py` 或逆向映射回物理动作空间 `action_space/` 用于驱动仿真环境或真实自车。
"""

import logging

import torch
from transformers import AutoTokenizer, StoppingCriteria

logger = logging.getLogger(__name__)


def to_special_token(token: str) -> str:
    """包装令牌（Token），将其转换为形如 `<|token|>` 的系统特殊Token格式。"""
    return "<|" + token + "|>"


def extract_traj_tokens(
    output_tokens: torch.Tensor,
    special_token_ids: dict[str, int],
    tokens_per_future_traj: int,
    future_token_start_idx: int,
    traj_tokenizer_vocab_size: int,
) -> torch.Tensor:
    """从网络预测生成的输出Token张量中，剥离出专属于轨迹预测的Tokens(高度向量化并行处理版本)。

    本函数基于矢量处理而不使用沿批次维度的For循环，能够在GPU上并行处理所有的Batch批次以加速。

    假设输出的Token排列形如: [...<|cot_end|> <|future_traj_start|>]<|future_traj|>...<|future_traj_end|>.

    参数:
        output_tokens (torch.Tensor): 语言模型生成的Token序列，形状 [B, L]。
        special_token_ids (dict[str, int]): 映射特殊标记(例如traj_future_start等)的ID字典。
        tokens_per_future_traj (int): 期望提取出来的标准未来预测轨迹Token数目。
        future_token_start_idx (int): 轨迹标记在整体自然语言词表(Vocab)中的起始偏置索引。
        traj_tokenizer_vocab_size (int): 分配给轨迹词汇表的最大专用词库容量大小。

    返回:
        torch.Tensor: 返回纯净的轨迹Token序列，形状将是 [B, tokens_per_future_traj]。
    """
    batch_size, seq_len = output_tokens.shape
    device = output_tokens.device

    # 初始化用于存放结果的预测输出序列张量，默认全0
    traj_tokens = torch.zeros(
        (batch_size, tokens_per_future_traj), dtype=output_tokens.dtype, device=device
    )

    # 针对每一个批次，寻找第一个出现的结束标记符(End Token)所在的位置
    # 如果找不到结束符，那么就假设到句尾整个都是有效的
    end_mask = output_tokens == special_token_ids["traj_future_end"]
    end_positions = torch.where(
        end_mask.any(dim=1),
        end_mask.int().argmax(dim=1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
    )

    # 针对每一个批次，寻找最后一个出现的开始符(Start Token)所在的位置
    # 为了找“最后一个”，我们将整个句尾反转寻找
    start_mask = output_tokens == special_token_ids["traj_future_start"]
    start_mask_reversed = torch.flip(start_mask, dims=[1])
    last_start_positions_reversed = start_mask_reversed.int().argmax(dim=1)
    start_positions = seq_len - 1 - last_start_positions_reversed
    start_positions = torch.where(
        start_mask.any(dim=1),
        start_positions,
        torch.full((batch_size,), -1, dtype=torch.long, device=device),
    )

    # 创建一个张量以广播生成范围（判断位置是否落在Start和End之间） [B, seq_len]
    range_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = (range_tensor > start_positions.unsqueeze(1)) & (
        range_tensor < end_positions.unsqueeze(1)
    )
    # 提取有效果位上的Token，并将无关位上的清零
    extracted_tokens = torch.where(valid_mask, output_tokens, torch.zeros_like(output_tokens))

    # 完整性自检：确认模型生成的轨迹Token个数是否匹配我们的预测目标长设定
    n_valid_tokens = valid_mask.sum(dim=1)
    mismatch_mask = n_valid_tokens != tokens_per_future_traj
    if mismatch_mask.any():
        for idx in mismatch_mask.nonzero(as_tuple=True)[0]:
            logger.warning(
                f"批次 {idx}: 提取到的Token数量与期望不符。 "
                f"预期: {tokens_per_future_traj}, 实际提取: {n_valid_tokens[idx].item()}."
            )

    # 只基于那些有效的标记位收集，且裁剪至我们指定的最大输出上限范围内
    cumsum_indices = torch.cumsum(valid_mask.int(), dim=1) - 1
    output_mask = valid_mask & (cumsum_indices < tokens_per_future_traj)

    if output_mask.any():
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        output_positions = cumsum_indices[output_mask]
        batch_ids = batch_indices[output_mask]
        token_values = extracted_tokens[output_mask]
        # 解除映射的偏置位（将词库中的Token ID倒车回专用数字空间 ID）
        token_values = token_values - future_token_start_idx

        # 检查是否含有无效越界的Token id
        invalid_tokens = (token_values < 0) | (token_values > traj_tokenizer_vocab_size)
        if invalid_tokens.any():
            logger.warning(f"在 {invalid_tokens.sum().item()} 个位置处发现非法的Token ID。")

        # 限位操作（Clamp），防止数组越界错误
        token_values = torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)
        traj_tokens[batch_ids, output_positions] = token_values

    return traj_tokens


def extract_between_special_tokens(decoded_batch: list[str], token: str) -> list[str]:
    """抽取自然语言文本中位于指定特殊成对Token区间里面的目标文本。

    参数:
        decoded_batch (list[str]): 我们要进行搜索和截取的源内容批次。
        token (str): 去掉 <| 和 |> 等包装后的目标特殊Token代号名。

    返回:
        list: 返回一列截取提纯过后的目标文本字符串。
    """
    start_token = to_special_token(f"{token}_start")
    end_token = to_special_token(f"{token}_end")

    out: list[str] = []
    apnd = out.append
    for s in decoded_batch:
        before_end, sep, _ = s.partition(end_token)
        if not sep:
            apnd("")
            continue
        i = before_end.rfind(start_token)
        if i != -1:
            apnd(before_end[i + len(start_token) :].strip())
        else:
            apnd(before_end.strip())
    return out


def extract_text_tokens(
    tokenizer: AutoTokenizer, output_tokens: torch.Tensor
) -> dict[str, list[str]]:
    """从VLM输出的Tokens中提取出附加生成的文本字段(例如自然语言回答'answer'、或者是模型思考层'cot'及元动作'meta_action')。

    参数:
        tokenizer: 用于解码的Token发生器(Tokenizer)对象。
        output_tokens (torch.Tensor): 形状为 [B*ns*nj, L] 的输出Tokens。

    返回:
        dict[str, list[str]]: 包含了包含所有有效文本序列数据的字典表。
    """
    # 将模型生成出的Token数组批次性解码为人类可读的字符串（保留其中的特殊控制标记）
    decoded_batch = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)

    # 预设模型可能的推理字段输出范围
    extract_tokens = ["cot", "meta_action", "answer"]
    extracted_text = {}
    for token in extract_tokens:
        extracted_text[token] = extract_between_special_tokens(decoded_batch, token)
    return extracted_text


class StopAfterEOS(StoppingCriteria):
    """
    这是一种动态停止规则类（StoppingCriteria）：
    其确保网络在预测出第一个 EOS（结束）Token 后，还能宽限生成一个额外的 Token，然后再强行中止推理。
    专门用于对接下一个模块缓存预处理操作时的特殊对齐需求。
    """

    def __init__(self, eos_token_id: int):
        """参数:
        eos_token_id (int): 设定的目标句子结束符Token ID。
        """
        self.eos_token_id = eos_token_id
        self.eos_found = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """模型推理每次产生字符调用的判定触发函数。

        参数:
            input_ids (torch.LongTensor): 当前已经累积生成的文本序列IDs，形状 [B, L]。
            scores (torch.FloatTensor): 概率分数映射。

        返回:
            bool: 返回True则意味着立即停止生成。
        """
        batch_size = input_ids.shape[0]

        # 如果是首次遇到判定函数，就初始化追踪标记
        if self.eos_found is None:
            self.eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if self.eos_found.all():
            return True

        # 判断最新生成出来的（最后一个也就是[:,-1]）Token是否等于我们寻找的EOS结束符标志
        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id

        # 将历史中已经出现过的情况和当前这一步汇总按位或(OR)整合缓存起来
        self.eos_found = self.eos_found | current_has_eos
        return False


def replace_padding_after_eos(
    token_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """找出各个序列中的第一结束符(EOS)，然后把自EOS后生成的多余Token全部强制替换为填充符(PAD)。

    参数:
        token_ids (torch.Tensor): 语言模型生成的Token标识形状 [B, L]。
        eos_token_id (int): 要寻找的结束标记位ID。
        pad_token_id (int): 用于遮罩填充标记位ID。

    返回:
        torch.Tensor: 被清理好的Token张量。
    """
    batch_size, seq_len = token_ids.shape

    # 找到每批中EOS在哪
    eos_mask = token_ids == eos_token_id  # [B, L]

    # 获取此标记第一次出现的坐标下标
    # 在没有EOS的地方添加seq_len长以处理没有EOS序列的情况
    eos_positions = torch.where(
        eos_mask,
        torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1),
        torch.tensor(seq_len, device=token_ids.device),
    )
    first_eos_pos = eos_positions.min(dim=1, keepdim=True)[0]  # [B, 1]

    # 创建掩码掩盖所有EOS之后的元素
    position_indices = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # [1, L]
    mask_after = position_indices > first_eos_pos  # [B, L]

    # 将其强制原地替换覆盖为 PAD token
    token_ids[mask_after] = pad_token_id
    return token_ids
    return token_ids
