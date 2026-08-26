# Alpamayo-R1 MoE 架构设计文档

## 1. 架构概述

将原有的单一 expert 模型升级为 Mixture-of-Experts (MoE) 架构：
- **3个轻量级 Expert**：结构与原 expert 相同，但使用更小的 Transformer decoder（层数和隐藏层维度减小）。
- **1个 Router**：利用 VLM 提取的 CoT hidden states 和 camera_front_wide_120fov 图像 hidden states 执行 top-1 hard routing。
- **Flow Matching 采样**：每个 expert 内部依然使用 flow matching 生成轨迹。
- **训练策略**：VLM 冻结，只训练 Router + 3个 Expert + action_in_proj + action_out_proj。
- **损失函数**：Flow Matching Loss + Load Balancing Loss。

## 2. 架构流程图

`mermaid
flowchart TD
    subgraph Frozen
        VLM[VLM - Qwen3-VL - Frozen]
    end

    Input[Input: Images + History Traj + Prompt] --> VLM
    VLM --> CoT_HS[CoT Hidden States]
    VLM --> Img_HS[Image Hidden States - camera_front_wide_120fov]
    VLM --> KV[KV Cache for Expert]

    CoT_HS --> Router
    Img_HS --> Router

    Router --> |top-1 gate| Expert_Select{Expert Selection}

    Expert_Select --> |expert_0| E0[Expert 0 - Lightweight Transformer]
    Expert_Select --> |expert_1| E1[Expert 1 - Lightweight Transformer]
    Expert_Select --> |expert_2| E2[Expert 2 - Lightweight Transformer]

    subgraph Each Expert
        ActionInProj[action_in_proj: noisy action + t -> embeddings]
        ExpertModel[Expert Transformer Decoder + KV Cache]
        ActionOutProj[action_out_proj: hidden -> velocity field]
        ActionInProj --> ExpertModel --> ActionOutProj
    end

    E0 --> FM[Flow Matching Sampling - Euler ODE]
    E1 --> FM
    E2 --> FM

    FM --> ActionSpace[Action Space -> Trajectory]
    ActionSpace --> Output[pred_xyz, pred_rot]
`

## 3. 算法流程对比

### 3.1 原始流程（单 Expert）

`	ext
VLM.generate() -> CoT tokens + KV cache
                        |
                        v
              step_fn(noisy_action, t):
                action_in_proj(x, t) -> token_embeds
                expert(token_embeds, KV_cache) -> hidden
                action_out_proj(hidden) -> velocity_field
                        |
                        v
              diffusion.sample(step_fn) -> sampled_action
                        |
                        v
              action_space.action_to_traj() -> pred_xyz, pred_rot
`

### 3.2 MoE 推理流程

`	ext
VLM.generate() -> CoT tokens + KV cache + hidden_states
                        |
                        v
              Router(cot_hidden_states, img_hidden_states) -> expert_idx
                        |
                        v
              step_fn(noisy_action, t):  # 使用 expert_idx 选中的 expert
                action_in_proj[expert_idx](x, t) -> token_embeds
                experts[expert_idx](token_embeds, KV_cache) -> hidden
                action_out_proj[expert_idx](hidden) -> velocity_field
                        |
                        v
              diffusion.sample(step_fn) -> sampled_action
                        |
                        v
              action_space.action_to_traj() -> pred_xyz, pred_rot
`

## 4. 关键模块设计

### 4.1 Router (Alpamayo_r1/models/router.py)

`python
class ExpertRouter(nn.Module):
    "`"`"
    输入：
      - cot_hidden: [B, L_cot, D_vlm]  VLM最后一层CoT token的hidden states
      - img_hidden: [B, L_img, D_vlm]  VLM最后一层camera_front_wide图像token的hidden states
    
    处理：
      1. 对 cot_hidden 做 mean pooling -> [B, D_vlm]
      2. 对 img_hidden 做 mean pooling -> [B, D_vlm]
      3. concat -> [B, 2*D_vlm]
      4. MLP -> [B, num_experts] logits
      5. top-1 argmax (推理) / Gumbel-Softmax (训练时保持可导)
    
    输出：
      - gate_logits: [B, num_experts]  用于计算 load balancing loss
      - expert_idx: [B]  选中的 expert 索引
    "`"`"
`

**关键设计点**：
- 训练时使用 **Straight-Through Gumbel-Softmax**，forward 走 hard one-hot，backward 走 soft gradient，保证梯度传回 router。
- 推理时直接 argmax。

### 4.2 轻量级 Expert

每个 expert 包含三个组件：
- expert: 小型 Transformer decoder，通过 expert_cfg 降低规模。
- action_in_proj: PerWaypointActionInProjV2
- action_out_proj: Linear

配置参数体现（通过 expert_cfg 控制）：
- 降低 num_hidden_layers（28 层 -> 8 层）
- 保持 hidden_size（3584，不缩宽度）
- 保持 intermediate_size（5120）
- 保持 num_attention_heads（28 头）

### 4.3 损失函数 (Alpamayo_r1/models/moe_loss.py)

#### Flow Matching Loss（每个 expert 内部计算）
与原始版本一致：
`	ext
L_fm = MSE(predicted_velocity_field, target_velocity_field)
`

其中 target velocity field 取 flow matching 直线插值路径的目标：
`	ext
x_t = (1-t) * noise + t * x_1
v_target = x_1 - noise
`

#### Load Balancing Loss
引入 Switch Transformer 的负载均衡损失，防止特定 expert 饿死或未被选择：

`	ext
L_balance = num_experts * sum_i(f_i * p_i)

其中：
  f_i = 第i个expert被选中的比例（当前batch内）
  p_i = router对第i个expert的平均gate概率（softmax后）
`

确保每个 expert 都能被均匀选择时，L_balance 最小。

#### 总损失
`	ext
L_total = L_fm + alpha * L_balance
`
alpha 为平衡系数（如 0.01~0.1）。

### 4.4 Config (Alpamayo_r1/config.py)

`python
class AlpamayoR1MoEConfig(AlpamayoR1Config):
    model_type = "alpamayo_r1_moe"
    
    def __init__(
        self,
        num_experts: int = 3,
        router_hidden_dim: int = 1024,
        balance_loss_weight: float = 0.01,
        expert_cfg: dict = None,  # 控制各个expert的规模
        **kwargs,
    ):
        ...
`

### 4.5 MoE 主模型 (Alpamayo_r1/models/alpamayo_r1_moe.py)

`python
class AlpamayoR1MoE(ReasoningVLA):
    "`"`"MoE版本的Alpamayo R1模型"`"`"
    
    # 初始化：
    #   - 冻结 VLM
    #   - 实例化 3 个 expert (小型 Transformer decoder)
    #   - 实例化 3 个 action_in_proj
    #   - 实例化 3 个 action_out_proj
    #   - 实例化 1 个 Router
    #   - 实例化 1 个 diffusion 采样器
    #   - 实例化 1 个 action_space
    
    # 推理逻辑 sample_trajectories_from_data_with_vlm_rollout:
    #   1. VLM generate -> CoT + KV cache
    #   2. 提取 CoT hidden states 和 img hidden states
    #   3. Router 选择 expert_idx
    #   4. 以选中的 expert 构造 step_fn
    #   5. diffusion.sample(step_fn) -> trajectory
    
    # 训练流程 forward:
    #   1. VLM forward (frozen) -> hidden_states
    #   2. Router -> expert_idx, gate_logits
    #   3. 选中的 expert 计算 flow matching loss
    #   4. 计算 load balancing loss
    #   5. 返回总损失
`

## 5. 提取 VLM Hidden States 的方法

### 5.1 CoT Hidden States

VLM generate 时使用 output_hidden_states=True，获取最后一层的 hidden states。CoT token 的位置可以通过 special token <|cot_start|> 和 <|cot_end|> 定位。

`python
# 在 vlm_outputs 中找到 cot 范围
cot_start_id = tokenizer.convert_tokens_to_ids('<|cot_start|>')
cot_end_id = tokenizer.convert_tokens_to_ids('<|cot_end|ข้อง>')
# 截取对应位置的 hidden states
`

### 5.2 Image Hidden States（camera_front_wide_120fov）

VLM 的 input_ids 中包含图像 token。camera_front_wide_120fov 对应的图像 token 位置可以通过 <|image_start|> 和 <|image_end|> 定位。由于 VLM 会在 generate 时缓存 prefill 阶段的 hidden states，可以从中提取。

**简化方案**：在 VLM forward/generate 之后，从 KV cache 或最后一层的 hidden states 中直接提取图像对应的值。

## 6. 文件结构设计

`	ext
alpamayo_r1/
├── config.py                    # 新增 AlpamayoR1MoEConfig
├── models/
│   ├── router.py                # 新增 ExpertRouter
│   ├── alpamayo_r1_moe.py       # 新增 AlpamayoR1MoE 主模型
│   ├── moe_loss.py              # 新增 load_balancing_loss
│   ├── alpamayo_r1.py           # 原有基线模型
│   ├── base_model.py            # 基础封装
│   ├── action_in_proj.py        # 动作投影
│   ├── ...
├── diffusion/
│   ├── flow_matching.py         # 流匹配相关
│   ├── ...
train_moe.py                     # MoE训练脚本
inference_moe.py                 # MoE推理脚本
`

## 7. 训练流程

`mermaid
flowchart TD
    Data[Training Data: images + history_traj + future_traj] --> VLM_FWD[VLM Forward - Frozen]
    VLM_FWD --> HS[Hidden States]
    VLM_FWD --> KV[KV Cache]
    
    HS --> Extract[Extract CoT + Img Hidden States]
    Extract --> Router[Router -> expert_idx, gate_logits]
    
    Router --> |expert_idx| FM_Train[Flow Matching Training]
    
    subgraph FM_Train[Flow Matching Training Step]
        GT[Ground Truth Action x1] --> Interp[x_t = 1-t x noise + t x x1]
        Noise[Sample noise x0] --> Interp
        T[Sample t ~ U 0 1] --> Interp
        Interp --> StepFn[Selected Expert: step_fn of x_t and t]
        StepFn --> Pred[Predicted velocity v]
        GT --> Target[Target velocity: x1 - noise]
        Pred --> L_FM[L_fm = MSE of v and v_target]
        Target --> L_FM
    end
    
    Router --> |gate_logits| L_BAL[L_balance = load balancing loss]
    
    L_FM --> L_TOTAL[L_total = L_fm + alpha x L_balance]
    L_BAL --> L_TOTAL
    
    L_TOTAL --> Backward[Backward - update Router + Experts + Projections]
`

## 8. 推理流程

`mermaid
flowchart TD
    Input[Input Data] --> VLM_GEN[VLM Generate - autoregressive CoT]
    VLM_GEN --> CoT_HS[CoT Hidden States]
    VLM_GEN --> Img_HS[Image Hidden States]
    VLM_GEN --> KV[KV Cache]
    
    CoT_HS --> Router[Router]
    Img_HS --> Router
    Router --> |top-1 argmax| IDX[expert_idx]
    
    IDX --> StepFn[Build step_fn with selected expert]
    KV --> StepFn
    
    StepFn --> Sample[diffusion.sample - Euler ODE]
    Sample --> Action[sampled_action]
    Action --> Traj[action_space.action_to_traj -> pred_xyz, pred_rot]
`

## 9. 关键实现细节

### 9.1 Straight-Through Gumbel-Softmax

训练时 Router 使用 ST Gumbel-Softmax 保证梯度可微：
`python
# forward: hard one-hot (argmax)
# backward: soft gradient (通过 Gumbel-Softmax回传)
y_soft = F.gumbel_softmax(logits, tau=1.0, hard=False)
idx = y_soft.argmax(dim=-1)
y_hard = F.one_hot(idx, num_experts).float()
gate = y_hard - y_soft.detach() + y_soft  # straight-through 技巧
`

### 9.2 Expert 选通的 Batch 处理

由于采用 hard routing，同一 batch 内不同样本可能选择不同 expert，实现方式：
`python
# 按 expert 索引遍历
for eid in range(num_experts):
    mask = (expert_idx == eid)
    if mask.any():
        # 只有选中该 expert 的样本才执行 forward
        output[mask] = experts[eid](input[mask], ...)
`

### 9.3 VLM 参数冻结

`python
for param in self.vlm.parameters():
    param.requires_grad = False
`

### 9.4 Hidden States 提取

推理时通过 output_hidden_states=True 获取 VLM 的 hidden states，需修改 vlm.generate() 调用参数。训练时通过 VLM forward 直接获取 hidden states（无需 generate）。

## 10. 实施步骤

1. 在 alpamayo_r1/config.py 中定义 AlpamayoR1MoEConfig
2. 新建 alpamayo_r1/models/router.py，实现 ExpertRouter
3. 新建 alpamayo_r1/models/moe_loss.py，实现 load_balancing_loss
4. 新建 alpamayo_r1/models/alpamayo_r1_moe.py，实现 AlpamayoR1MoE 架构
5. 实现 inference_moe.py 进行 MoE 预测推理
6. 实现 train_moe.py 开展训练流程
