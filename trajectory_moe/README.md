# Alpamayo-R1 MoE: 基于混合专家架构的轨迹预测视觉语言模型 🏎️🧠

## 🌟 项目简介 (Introduction)
本项目是我在实习期间主导的一项科研探索，旨在优化自动驾驶场景下的端到端轨迹预测模块。
原有架构采用单一庞大的 Transformer Decoder 作为“专家”来生成轨迹，本研究提出将其升级为 **Mixture-of-Experts (MoE)** 架构。通过引入轻量级专家集群与多模态视觉-语言特征耦合的路由机制（Router），我们在保持以及提升轨迹预测精度的同时，显著降低了推理延迟，增强了模型在复杂长尾场景下的泛化与调度能力。

## 💡 科研思路与动机 (Research Motivation)
在传统的 VLM（视觉语言模型）端到端自动驾驶中，单一的轨迹生成头部难以同时兼顾所有场景（如：红绿灯启停、跟车、复杂路口转向）。
*   **思路1：术业有专攻**。引入 MoE 机制，将单一庞大的 Expert 拆分为 3 个轻量级的 Transformer Decoder (Layers: 28 -> 8, Hidden size: 3584 -> 1024)。让不同的专家去专门拟合特定类型的行驶行为（如跟车专家、停止专家、转向专家）。
*   **思路2：多模态特征路由 (Multimodal Routing)**。设计了一个 Router 模块，创新性地将 VLM 生成的 CoT (Chain-of-Thought) 隐藏状态与前视广角相机的图像特征 (camera_front_wide_120fov) 结合起来，进行 Top-1 的硬件路由选择，使专家调度与多模态语义强绑定。
*   **思路3：高质量轨迹流匹配**。在每个轻量级专家内部，保持 Flow Matching 生成范式（经由 Euler ODE 采样），以确保平滑且符合动力学的轨迹输出。

## 🛠️ 网络架构设计 (Architecture Design)
项目核心架构流程如下：
1. **Frozen VLM (Qwen3-VL)**: 冻结 VLM 的主干网络，通过输入图像、历史轨迹与 Prompt，生成 CoT 思维链及其隐层特征。
2. **Multimodal Router**: 对 CoT 隐藏特征与图像特征进行 Pooling 和 Concat，通过 MLP 预测并利用 Straight-Through Gumbel-Softmax 选择对应的 Expert 编号。
3. **MoE Layer**: 
    *   3 个独立平行的轻量级 Expert。
    *   使用选择到的 `expert_idx` 定向激活对应的 `action_in_proj` 和 `action_out_proj`。
4. **Flow Matching Sampling**: 基于选中的 Expert 输出预测 Target Velocity Field，并沿时间步 $t$ 执行 Euler ODE 采样，解码出预测轨迹 (`pred_xyz`, `pred_rot`)。

> **优化目标 (Loss Function)**：
> $L_{total} = L_{FM} (\text{流匹配MSE}) + \alpha \cdot L_{balance} (\text{负载均衡Loss})$
> *引入 Switch Transformer 的 Load Balancing Loss，防止特定专家训练过载或缺乏更新。*

## � 核心架构创新：定制化的多模态 Macro-MoE (Core Innovations)
本架构在底层逻辑上借用了经典的稀疏激活 MoE 架构（类似于 Switch Transformer），但并未简单套用，而是针对“端到端自动驾驶”和“多模态大模型”场景做了极致的 **定制化（Macro-MoE / 任务级 MoE）** 改造。主要体现在以下三个层次：

### 1. 极速推理的路由机制 (Top-1 Hard Routing)
*   **做法**：区别于 Top-2 的 Mixtral 或 Soft-MoE，我们的 Router 选用了 **Top-1 路由（Hard Routing）**，每次仅激活 1 个专家，并引入了 Switch Transformer 中经典的 **Load Balancing Loss（负载均衡损失）** 防止专家“饿死”。
*   **科研动机**：在车端部署中，**推理延迟（Latency）是第一指标**。Top-1 路由在推理时只激活 $\frac{1}{3}$ 的参数，最大化地削减了显存峰值和计算开销；训练时则采用 **Straight-Through Gumbel-Softmax** 技巧，既保证了离散的路由选择（one-hot），又能让梯度平滑回传。

### 2. 专家定义：宏观任务解码器 (Macro-MoE vs Token-level FFN)
*   **标准 LLM 的 MoE**：如 Qwen、GPT-4 等模型，它们的专家通常局限于 Transformer 内部每个 Block 的 **FFN（前馈神经网络）层**。模型在生成每一个 Token 时极速切换不同的 FFN。
*   **我们的创新**：本项目设计的是一个 **宏观级别（Macro-level）** 的 MoE。“专家”并非零碎的 FFN 层，而是 **一整个完全独立的轻量级 Transformer Decoder（包含 8 层 Attention + Flow Matching 预测头部）**。专家围绕 **“宏观驾驶行为”**（如：跟车、启停、避让）构建。这也正是项目中包含大量 `cluster_*.py` 聚类脚本的原因——我们利用运动学物理特征先验，显式地指导了专家的分工。

### 3. 多模态语义融合：可解释的 Router 输入 (Multimodal Semantic Routing)
*   **标准 MoE**：Router 输入通常仅依赖上一层纯文本的隐层特征。
*   **我们的创新**：Router 的输入实现了 **多模态深度拼接**。我们将冻结的预训练大模型 (Qwen3-VL) 输出的 **思维链特征（CoT Hidden States）** 与 **前视相机的图像特征（Image Hidden States）** 进行融合（Mean Pooling -> Concat -> MLP）。
*   **科研价值**：赋予了模型极具解释性的物理映射能力。Router 不仅“看着”物理画面做决定，还“听着”大模型的逻辑推理（例如明确提示 "Stop for the red straight traffic light"），从而指派“刹车专家”来生成轨迹。

> **🌟 一句话总结（Resume Highlight）**：
> *本项目底层参考了 Switch Transformer 的 Top-1 稀疏门控思想以追求极致的端侧推理加速。在架构层面，我提出了一种多模态特征（图像语义+大模型思维链）深度耦合的 Macro-MoE 范式，将轨迹生成的任务解耦给多个具备物理驾驶行为先验的轻量级 Transformer Decoder 专家。当前版本通过路由准确率 95.80% 验证了该范式的可行性，同时在推理延迟（↓43.75%）、显存需求（↓55%）、参数激活率（↓66.9%）等方面实现了显著优化。精度优化是下一阶段的重点工作。*

## �📊 实验评估与效果 (Experiments & Results)
在项目中，我构建了完整的评测流水线，从精度和性能两方面论证了设计的有效性：

1. **路由准确率 (Routing Accuracy)**
   * 在全量测试集（涵盖近一万个轨迹片段）的 MoE 评估结果（`eval_moe_results.csv`）表明，该模型 **Router 命中准确率高达 95.8%**。Router 能够精准根据思维链意图指令指派对应的停车、减速或转向专家。
2. **轨迹预测精度 (Min-ADE / 初步探索阶段)**
   * 这是一项初步探索研究。当前阶段，MoE 架构在轨迹精度上相比 R1 单一 Decoder 基线有优化空间。
   * 数据验证：在评估集上，R1 单一 Decoder 基线的轨迹最小平均位移误差（Min-ADE）为 **0.8652m**（2,805 样本），而在 MoE 架构下，该误差为 **1.1134m**（9,620 样本），目前呈现 **28.69% 的精度降低**。虽然精度需要优化，但这反映了初步探索的真实状态。
   * **关键优势**：虽然精度有改进空间，但 Router 准确率达 **95.80%**，证明多模态路由策略的有效性。同时推理延迟显著下降，显存需求大幅降低，为后续优化奠定了坚实基础。
3. **轻量化与资源利用率**
   * 尽管当前精度有优化空间，但轻量化设计取得了显著成效。拆分出的 3 个专家极度轻量化（隐藏层维度从 3584 降低至 1024，层数从 28 降至 8），MoE 仅激活单专家时的轨迹生成解码耗时从 **3.2s 降至 1.8s**（下降 **43.75%**），显存峰值从 **40GB+ 降至 18GB+**（下降 **55%**），激活参数从 10B 降至 3.3B（仅占 1/3）。这在计算资源控制和端侧部署可行性上有极大的改善。
   * **可探索空间**：精度优化可通过细化聚类粒度（K=5+）、软路由或 Top-2 混合、改进专家初始化等方向实现。

## 📂 核心代码结构与职责 (Repository Structure)

项目主要划分为以下几个功能模块，详情见各代码文件头部注释：

- **数据分析与聚类 (`cluster_*.py`, `kmeans_plan.py`)**：通过运动学和物理特征对真实轨迹进行 K-Means 聚类，引导专家的倾向性标定。
- **训练流程 (`train_*.py`)**：包含了路由器（Router）、单专家（Single Expert）、聚合版大专家（Decoder Expert）及混合专家架构（MoE Finetune）的训练与微调脚本，支持多卡通信（_m版）。
- **测评流水线 (`eval_*.py`)**：针对生成精度（ADE偏差）、端到端采样以及模块耗时（Timing）的全方位验证代码。
- **推理部署 (`inference_*.py`)**：涵盖 MoE 的前向在线推理以及结合 VQA 思维链预测的主要逻辑验证。