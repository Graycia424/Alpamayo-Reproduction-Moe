#!/usr/bin/env python3
"""专家 / 路由器参数量解析计算脚本。

模块用途
    README 中引用的参数量均由本脚本产生，便于任何人复算，无需下载 checkpoint。

设计原则
    轻量专家「保持原始宽度，只减深度」：
    - hidden_size 保持原始 3584（Qwen2.5-VL-7B 主线，不缩到 1024）
    - num_hidden_layers 从 28 减到 8
    - intermediate_size 保持原始（5120）
    - 参数节省不靠缩宽度，而靠 Top-1 稀疏激活：每次推理只激活 3 个专家中的 1 个。

来源与依据
    1. 主干 Cosmos-Reason1-7B，由 Qwen2.5-VL-7B-Instruct 初始化（HF 标签 base_model=Qwen2.5-VL-7B）
       Qwen2.5-VL-7B：28 层，hidden 3584，28 个注意力头，4 个 KV 头（GQA），head_dim 128。
    2. 官方 Alpamayo-R1 单专家约 2.3B（论文/HF 模型卡）。

注意
    统计不含 action_in_proj / action_out_proj（每专家约 1M 量级），
    也不含 embed_tokens —— 后者在 alpamayo_r1.py:94 被显式删除。
"""

from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """Qwen 风格解码器专家的结构参数。"""

    name: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int


def count_layer_params(cfg: ExpertConfig) -> dict[str, int]:
    """计算单个解码器层的参数量，返回各组件名到参数量的字典。"""
    d = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim  # 查询投影输出维度
    kv_dim = cfg.num_key_value_heads * cfg.head_dim  # GQA 下 K/V 投影输出维度
    parts = {
        "self_attn.q_proj": d * q_dim,
        "self_attn.k_proj": d * kv_dim,
        "self_attn.v_proj": d * kv_dim,
        "self_attn.o_proj": q_dim * d,
        "self_attn.q_norm": cfg.head_dim,
        "self_attn.k_norm": cfg.head_dim,
        "mlp.gate_proj": d * cfg.intermediate_size,
        "mlp.up_proj": d * cfg.intermediate_size,
        "mlp.down_proj": cfg.intermediate_size * d,
        "input_layernorm": d,
        "post_attention_layernorm": d,
    }
    parts["total"] = sum(parts.values())
    return parts


def count_expert_params(cfg: ExpertConfig) -> int:
    """计算整个专家的参数量（含末尾的 expert.norm）。"""
    return count_layer_params(cfg)["total"] * cfg.num_hidden_layers + cfg.hidden_size


def count_router_params(vlm_hidden: int, router_hidden: int, num_experts: int) -> int:
    """计算路由器参数量。

    结构见 alpamayo_r1/models/router.py:
        Linear(vlm_hidden*2 -> router_hidden) + SiLU + Linear(router_hidden -> num_experts)
    """
    layer1 = vlm_hidden * 2 * router_hidden + router_hidden
    layer2 = router_hidden * num_experts + num_experts
    return layer1 + layer2


# 主干为 Cosmos-Reason1-7B（Qwen2.5-VL-7B-Instruct 初始化）：hidden 3584，28 层
VLM_HIDDEN = 3584
# 官方 Alpamayo-R1 单专家约 2.3B（论文/HF 模型卡）
OFFICIAL_EXPERT_PARAMS = 2_300_000_000

LIGHT = ExpertConfig(
    name="轻量专家 (保持宽度 3584，只减层数到 8)",
    hidden_size=3584,          # 保持原始宽度
    intermediate_size=5120,    # 保持原始中间层
    num_attention_heads=28,    # Qwen2.5-VL-7B 的 28 头
    num_key_value_heads=4,     # GQA 4 个 KV 头
    head_dim=128,
    num_hidden_layers=8,       # 只减层数 28 -> 8
)


def main() -> None:
    """打印轻量专家参数量、MoE 汇总，以及与官方约 2.3B 的稀疏激活对比。"""
    cfg = LIGHT
    parts = count_layer_params(cfg)
    total = count_expert_params(cfg)
    print(f"\n{cfg.name}")
    print(f"  hidden={cfg.hidden_size}  intermediate={cfg.intermediate_size}  "
          f"heads={cfg.num_attention_heads}  kv_heads={cfg.num_key_value_heads}  "
          f"layers={cfg.num_hidden_layers}")
    for k, v in parts.items():
        if k != "total":
            print(f"    {k:<28} {v:>15,}")
    print(f"    {'单层合计':<26} {parts['total']:>15,}")
    print(f"    {'单个专家':<26} {total:>15,}  ({total / 1e6:.1f} M)")

    router = count_router_params(VLM_HIDDEN, router_hidden=1024, num_experts=3)

    print("\nMoE 汇总（3 专家 + 路由器）")
    print(f"  单个轻量专家              {total:>15,}  ({total / 1e6:.1f} M)")
    print(f"  3 个专家合计（存储）      {total * 3:>15,}  ({total * 3 / 1e9:.2f} B)")
    print(f"  路由器                    {router:>15,}  ({router / 1e6:.2f} M)")
    print(f"  总存储                    {total * 3 + router:>15,}  "
          f"({(total * 3 + router) / 1e9:.2f} B)")
    print(f"\n  对比官方单专家 {OFFICIAL_EXPERT_PARAMS / 1e9:.2f} B（论文/HF 模型卡）：")
    print(f"    Top-1 每次只激活 1 专家  {total / OFFICIAL_EXPERT_PARAMS * 100:.1f}%  "
          f"（参数节省来自稀疏激活，而非缩宽度）")


if __name__ == "__main__":
    main()
