"""MoE variant of Alpamayo R1 ?C 3 lightweight experts with a learned router."""

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.config import AlpamayoR1MoEConfig
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.base_model import ReasoningVLA, SPECIAL_TOKENS
from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
from alpamayo_r1.models.router import ExpertRouter
from alpamayo_r1.models.moe_loss import load_balancing_loss
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class AlpamayoR1MoE(ReasoningVLA):
    """Mixture-of-Experts variant: 3 lightweight experts + router."""

    config_class = AlpamayoR1MoEConfig
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1MoEConfig,
        pretrained_modules: dict[str, nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        num_experts = config.num_experts
        expert_text_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for k, v in config.expert_cfg.items():
                setattr(expert_text_config, k, v)

        # Shared modules
        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg, x_dims=self.action_space.get_action_space_dims()
        )

        # Per-expert modules
        experts, in_projs, out_projs = [], [], []
        for _ in range(num_experts):
            expert = AutoModel.from_config(copy.deepcopy(expert_text_config))
            del expert.embed_tokens
            experts.append(expert)
            in_projs.append(
                hyu.instantiate(
                    config.action_in_proj_cfg,
                    in_dims=self.action_space.get_action_space_dims(),
                    out_dim=expert_text_config.hidden_size,
                )
            )
            out_projs.append(
                hyu.instantiate(
                    config.action_out_proj_cfg,
                    in_features=expert_text_config.hidden_size,
                    out_features=self.action_space.get_action_space_dims()[-1],
                )
            )
        self.experts = nn.ModuleList(experts)
        self.action_in_projs = nn.ModuleList(in_projs)
        self.action_out_projs = nn.ModuleList(out_projs)

        # Router
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        self.router = ExpertRouter(vlm_hidden_dim, config.router_hidden_dim, num_experts)

        # Dtype alignment
        if config.keep_same_dtype:
            dtype = self.experts[0].dtype
            for m in [self.diffusion, *self.action_in_projs, *self.action_out_projs, self.router]:
                m.to(dtype=dtype)

        # Freeze VLM
        for p in self.vlm.parameters():
            p.requires_grad = False

        self.post_init()

    # ------------------------------------------------------------------
    # Hidden-state extraction helpers
    # ------------------------------------------------------------------
    def _extract_hidden_ranges(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor, start_id: int, end_id: int
    ) -> torch.Tensor:
        """Extract hidden states between start_id and end_id tokens, mean-pooled per sample."""
        B, L, D = hidden_states.shape
        pooled = torch.zeros(B, D, device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(B):
            starts = (token_ids[i] == start_id).nonzero(as_tuple=True)[0]
            ends = (token_ids[i] == end_id).nonzero(as_tuple=True)[0]
            if len(starts) > 0 and len(ends) > 0:
                s, e = starts[0].item() + 1, ends[0].item()
                if e > s:
                    pooled[i] = hidden_states[i, s:e].mean(dim=0)
        return pooled.unsqueeze(1)  # [B, 1, D] to keep interface consistent

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
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
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B = ego_history_xyz.shape[0]
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) VLM autoregressive generation with hidden states
        max_generation_length = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
        gen_cfg = self.vlm.generation_config
        gen_cfg.top_p = top_p
        gen_cfg.temperature = temperature
        gen_cfg.do_sample = True
        gen_cfg.num_return_sequences = num_traj_samples
        gen_cfg.max_new_tokens = max_generation_length
        gen_cfg.output_logits = True
        gen_cfg.output_hidden_states = True
        gen_cfg.return_dict_in_generate = True
        gen_cfg.top_k = top_k
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id

        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList([
            ExpertLogitsProcessor(
                traj_token_offset=self.config.traj_token_start_idx,
                traj_vocab_size=self.config.traj_vocab_size,
            )
        ])
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=gen_cfg,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]

        # 2) Extract hidden states for router
        # Collect last-layer hidden states from decoder steps
        # hidden_states is a tuple of (num_steps, ) each (num_layers, [B, 1, D])
        last_layer_hs = []
        for step_hs in vlm_outputs.hidden_states:
            last_layer_hs.append(step_hs[-1])  # last layer, [B, seq_chunk, D]
        all_hidden = torch.cat(last_layer_hs, dim=1)  # [b_star, L_generated, D]

        # Build full sequence token ids for locating CoT / image ranges
        seq_ids = vlm_outputs.sequences  # [b_star, L_total]

        cot_start_id = self.special_token_ids.get("cot_start")
        cot_end_id = self.special_token_ids.get("cot_end")
        img_start_id = self.special_token_ids.get("image_start")
        img_end_id = self.special_token_ids.get("image_end")

        # For generated hidden states, offset = input_ids length
        input_len = input_ids.shape[1]
        # CoT tokens are in the generated part; image tokens are in the prefill part.
        # We use the generated hidden states for CoT, and approximate image features
        # from the first few hidden states (prefill portion captured in first step).
        gen_token_ids = seq_ids[:, input_len:]  # generated token ids

        # CoT hidden: from generated hidden states
        cot_hidden = self._extract_hidden_ranges(all_hidden, gen_token_ids, cot_start_id, cot_end_id)

        # Image hidden: use prefill hidden states (first step contains prefill)
        prefill_hs = vlm_outputs.hidden_states[0][-1]  # [b_star, prefill_len, D]
        prefill_ids = seq_ids[:, :input_len]
        img_hidden = self._extract_hidden_ranges(prefill_hs, prefill_ids, img_start_id, img_end_id)

        # Router decision
        gate_logits, expert_idx = self.router(cot_hidden, img_hidden)
        # For multi-sample: all samples from same input use same expert
        # expert_idx shape: [b_star] where b_star = B * num_traj_samples

        # 3) Build step_fn per expert and run diffusion
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full((b_star,), vlm_outputs.sequences.shape[1] - 1, device=device)
        valid_token_pos_id = torch.where(has_traj_future_start, traj_future_start_positions, last_token_positions)
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
            dtype=torch.float32, device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i]:-n_diffusion_tokens] = torch.finfo(attention_mask.dtype).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            pred = torch.zeros_like(x)
            for eid in range(self.config.num_experts):
                mask = expert_idx == eid
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                x_e = x[idx]
                t_e = t.expand(b, *([1] * (x.dim() - 1)))[idx]
                embeds = self.action_in_projs[eid](x_e, t_e)
                if embeds.dim() == 2:
                    embeds = embeds.view(idx.shape[0], n_diffusion_tokens, -1)
                out = self.experts[eid](
                    inputs_embeds=embeds,
                    position_ids=position_ids[:, idx],
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask[idx],
                    use_cache=True,
                    **forward_kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                h = out.last_hidden_state[:, -n_diffusion_tokens:]
                pred[idx] = self.action_out_projs[eid](h).view(-1, *self.action_space.get_action_space_dims())
            return pred

        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}
        sampled_action = self.diffusion.sample(
            batch_size=total_batch, step_fn=step_fn, device=device,
            return_all_steps=False, **diffusion_kwargs,
        )

        hist_xyz_rep = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        hist_rot_rep = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)

        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            for k in extra:
                extra[k] = np.array(extra[k]).reshape([input_ids.shape[0], num_traj_sets, num_traj_samples])
            extra["expert_idx"] = expert_idx.cpu().numpy()
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        traj_data: dict[str, Any],
        gt_actions: torch.Tensor,
        **vlm_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            input_ids: [B, L] tokenized input.
            attention_mask: [B, L].
            traj_data: dict with ego_history_xyz/rot.
            gt_actions: [B, *action_dims] ground-truth actions from flow matching.
            **vlm_kwargs: extra inputs for VLM (pixel_values, etc.).

        Returns:
            dict with 'loss', 'fm_loss', 'balance_loss', 'expert_idx'.
        """
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        B = input_ids.shape[0]
        device = input_ids.device

        # VLM forward (frozen) to get hidden states + KV cache
        with torch.no_grad():
            vlm_out = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                **vlm_kwargs,
            )
        last_hidden = vlm_out.hidden_states[-1]  # [B, L, D]
        full_kv_cache = vlm_out.past_key_values

        # Extract CoT and image hidden states
        cot_start_id = self.special_token_ids.get("cot_start")
        cot_end_id = self.special_token_ids.get("cot_end")
        img_start_id = self.special_token_ids.get("image_start")
        img_end_id = self.special_token_ids.get("image_end")

        cot_hidden = self._extract_hidden_ranges(last_hidden, input_ids, cot_start_id, cot_end_id)
        img_hidden = self._extract_hidden_ranges(last_hidden, input_ids, img_start_id, img_end_id)

        # Router
        gate_logits, expert_idx = self.router(cot_hidden, img_hidden)

        # Flow matching loss per expert
        t = torch.rand(B, 1, 1, device=device)
        noise = torch.randn_like(gt_actions)
        x_t = (1 - t) * noise + t * gt_actions
        v_target = gt_actions - noise

        # Predict velocity field using selected experts
        v_pred = torch.zeros_like(gt_actions)
        for eid in range(self.config.num_experts):
            mask = expert_idx == eid
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]

            # Slice KV cache for this expert's samples
            kv_cache_sliced = DynamicCache(
                ddp_cache_data=[(k[idx], v[idx]) for k, v in full_kv_cache]
            )

            n_diff = self.action_space.get_action_space_dims()[0]
            embeds = self.action_in_projs[eid](x_t[idx], t[idx].squeeze(-1))
            if embeds.dim() == 2:
                embeds = embeds.view(idx.shape[0], n_diff, -1)

            expert_out = self.experts[eid](
                inputs_embeds=embeds,
                past_key_values=kv_cache_sliced,
                use_cache=False,
            )
            h = expert_out.last_hidden_state[:, -n_diff:]
            v_pred[idx] = self.action_out_projs[eid](h).view(-1, *self.action_space.get_action_space_dims())

        fm_loss = torch.nn.functional.mse_loss(v_pred, v_target)
        bal_loss = load_balancing_loss(gate_logits, expert_idx, self.config.num_experts)
        loss = fm_loss + self.config.balance_loss_weight * bal_loss

        return {
            "loss": loss,
            "fm_loss": fm_loss.detach(),
            "balance_loss": bal_loss.detach(),
            "expert_idx": expert_idx.detach(),
        }


AutoConfig.register("alpamayo_r1_moe", AlpamayoR1MoEConfig)
AutoModel.register(AlpamayoR1MoEConfig, AlpamayoR1MoE)
