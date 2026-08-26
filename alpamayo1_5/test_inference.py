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
主要功能：一个最简化的用于执行和测试端到端推理数据流水线的小型用例入口脚本 (`test_inference.py`)。
数据流与模块关系说明：
1. 【输入来源】：依赖 `load_physical_aiavdataset.py` 从本地或远程接口加载和拼装多模态视频与运动学张量；引入测试级别的单一 `clip_id` 数据片段。
2. 【核心处理】：加载 `Alpamayo1_5` 模型后，将多模态数据输入主模型，并利用 `helper.py` 辅助生成标准计算格式，获取自回归采样的动作输出。
3. 【输出去向】：最终对比预测的 3D 轨迹与源数据的真实标签 (Ground Truth) 位置差，计算出 `minADE`（最小平均位移误差）并将结果通过标准输出打印在终端。
"""

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def main() -> None:
    """Run inference on an example clip and report minADE."""
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    if min_ade >= 1.0:
        print(f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic.")


if __name__ == "__main__":
    main()
