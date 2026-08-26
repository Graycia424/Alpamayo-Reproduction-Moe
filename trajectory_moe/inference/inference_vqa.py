# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 1.5 的多模态视觉问答验证模块。
主要功能：专门验证原版模型处理跨模态场景描述、解读交通元素等 VQA（视觉问答）的能力测试脚本。
注意：这是这三个文件中，唯一一个沿用 Alpamayo 1.5 基线的实验脚本（主要测试原生的基础问答能力）。

【和其他文件的依赖调用关系】
向上依赖（调了谁）：
- 依赖 `alpamayo1_5.models.alpamayo1_5` 原生版本基座模型。
- 依赖 `alpamayo1_5.load_physical_aiavdataset` 数据读取封装。
"""

#!/usr/bin/env python3

import mediapy as mp
import pandas as pd
import physical_ai_av

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper


def main():
    """Main function to run VQA inference with Alpamayo 1.5 model"""
    
    print("=" * 80)
    print("Alpamayo 1.5: VQA Inference")
    print("=" * 80)
    
    # ============================================
    # Load model and construct data preprocessor
    # ============================================
    print("\n[1/5] Loading model...")
    model = Alpamayo1_5.from_pretrained(
        "/path/to/data/Alpamayo-1.5-10B", 
        dtype=torch.bfloat16
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("✓ Model loaded successfully")
    
    # ============================================
    # Load and prepare data
    # ============================================
    print("\n[2/5] Loading dataset...")
    clip_index = pd.read_parquet("/path/to/data/PhysicalAI-Autonomous-Vehicles/clip_index.parquet")
    clip_ids = clip_index.index.tolist()
    clip_id = clip_ids[774]
    
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data = load_physical_aiavdataset(
        clip_id,
        camera_features=[
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        ],
    )
    print(f"✓ Loaded clip: {clip_id}")
    print(f"  Image frames shape: {data['image_frames'].shape}")
    print(f"  Camera indices: {data['camera_indices']}")
    
    # ============================================
    # Visualize the video (optional)
    # ============================================
    print("\n[3/5] Visualizing video frames...")
    frames = data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1)
    mp.show_images(frames, columns=4, width=200)
    print("✓ Video frames displayed")
    
    # ============================================
    # First question: Describe the scene
    # ============================================
    print("\n[4/5] Running inference on first question...")
    question1 = "Describe the scene."
    
    # Create message for first question
    messages1 = helper.create_vqa_message(
        data["image_frames"].flatten(0, 1),
        question=question1,
        camera_indices=data["camera_indices"],
    )
    
    # Prepare inputs
    inputs1 = processor.apply_chat_template(
        messages1,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"  Sequence length: {inputs1.input_ids.shape}")
    
    model_inputs1 = {"tokenized_data": inputs1}
    model_inputs1 = helper.to_device(model_inputs1, "cuda")
    
    # Run inference
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        extra1 = model.generate_text(
            data=model_inputs1,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_generation_length=256,
        )
    
    print("\n" + "=" * 50)
    print(f"Question 1: {question1}")
    print("-" * 50)
    print(f"Answer 1: {extra1['answer'][0]}")
    print("=" * 50)
    
    # ============================================
    # Second question: Traffic elements and driving behavior
    # ============================================
    print("\n[5/5] Running inference on second question...")
    question2 = "What are the key traffic elements visible in this scene and how should they influence driving behavior?"
    
    # Create message for second question
    messages2 = helper.create_vqa_message(
        data["image_frames"].flatten(0, 1),
        question=question2,
        camera_indices=data["camera_indices"],
    )
    
    # Prepare inputs
    inputs2 = processor.apply_chat_template(
        messages2,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs2 = {"tokenized_data": inputs2}
    model_inputs2 = helper.to_device(model_inputs2, "cuda")
    
    # Run inference
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        extra2 = model.generate_text(
            data=model_inputs2,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_generation_length=256,
        )
    
    print("\n" + "=" * 50)
    print(f"Question 2: {question2}")
    print("-" * 50)
    print(f"Answer 2: {extra2['answer'][0]}")
    print("=" * 50)
    
    print("\n✓ Inference completed successfully!")


if __name__ == "__main__":
    main()