import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import T5PreTrainedModel
from .modeling_t5_instance_rope import T5Stack as T5Stack_rope
from transformers.utils import ModelOutput

from .utils import InstanceNorm, Patch, get_log_decay_weights
from .patch_utils import _divide_patches, _create_initial_setup, _create_initial_position_mapping, \
    _update_parent_mapping, _update_position_mapping, _map_to_parent_blocks, _create_granularity_mask, \
    _generate_x_final
from .layers import ResidualBlock, MultiInResidualBlock
from .moe import ModelArgs, MoE
from .configuration_kairos import KairosConfig


@dataclass
class KairosOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    prediction_outputs: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None
    future_target: Optional[torch.Tensor] = None


class DynamicPatch(nn.Module):
    def __init__(
            self,
            max_patch_size: int,
            patch_stride: int,
            levels: int,
            n_null_experts: int = 0,
            n_activated_experts: int = 1,
            moe_inter_dim: int = 1408,
            update_bias_rate: float = 0.001,
            target_dist: list = None,
            route_scale: float = 1.0,
    ):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.patch_stride = patch_stride
        self.levels = levels
        self.patch = Patch(max_patch_size, patch_stride)
        args = ModelArgs()
        args.dim = max_patch_size
        args.n_real_experts = levels
        args.n_null_experts = n_null_experts
        args.n_routed_experts = args.n_real_experts + args.n_null_experts
        args.n_activated_experts = n_activated_experts
        args.moe_inter_dim = moe_inter_dim
        args.update_bias_rate = update_bias_rate
        args.target_dist = target_dist
        args.route_scale = route_scale
        self.moe = MoE(args)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            x: [1, 8] = [[ 1.5550, -0.0704,  0.4447, -0.4042,  0.0302, -0.1819,  0.0508,  0.4408]]
            mask: [1, 8] = [[1., 1., 1., 1., 1., 1., 1., 1.]]
        output:
            patched_x: [1, 3, 4] = [[[ 1.5550, -0.0704,  0.0000,  0.0000],
                                [ 0.4447, -0.4042,  0.0000,  0.0000],
                                [ 0.0302, -0.1819,  0.0508,  0.4408]]]
            patched_mask: [1, 3, 4] = [[[1., 1., 0., 0.],
                                    [1., 1., 0., 0.],
                                    [1., 1., 1., 1.]]]
            size: [1, 3] = [[2, 2, 4]]
        """
        patched_x, patched_mask = self.patch(x), self.patch(mask)
        size = torch.full(patched_x.shape[:-1], self.max_patch_size, dtype=torch.int64, device=x.device)
        patched_x, patched_mask, size, weights, indices, x_final = self._divide_patches_by_moe(patched_x, patched_mask,
                                                                                               size)
        return patched_x, patched_mask, size, weights, indices, x_final

    def _divide_patches_by_moe(
            self, x: torch.Tensor, mask: torch.Tensor, size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input:
            x: [1, 2, 4] = [[[1.5550, -0.0704,  0.4447, -0.4042], [0.0302, -0.1819,  0.0508,  0.4408]]]
            mask: [1, 2, 4] = [[[1., 1., 1., 1.], [1., 1., 1., 1.]]]
            size: [1, 2] = [[4, 4]]
            weights: [1, 2, 2] = [[[0,6, 0.3], [0.5, 0.3]]]
            indices: [1, 2, 2] = [[[1, 0], [1, 2]]]
        output:
            new_x: [1, 6, 4] = [[[ 1.5550, -0.0704,  0.0000,  0.0000],
                                [ 0.4447, -0.4042,  0.0000,  0.0000],
                                [ 0.0302, 0.0000,  0.0000,  0.0000],
                                [ -0.1819, 0.0000,  0.0000,  0.0000],
                                [ 0.0508, 0.0000,  0.0000,  0.0000],
                                [ 0.4408, 0.0000,  0.0000,  0.0000]]]
            new_mask: [1, 6, 4] = [[[1., 1., 0., 0.],
                                    [1., 1., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [1., 0., 0., 0.]]]
            new_size: [1, 6] = [[2, 2, 1, 1, 1, 1]]
            weights: [1, 6, 2] = [[[0,6, 0.3], [0,6, 0.3], [0.5, 0.3], [0.5, 0.3], [0.5, 0.3], [0.5, 0.3]]]
            indices: [1, 6, 2] = [[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]]
            parent_blocks: [1, 6, 4] = [[[ 1.5550, -0.0704,  0.4447, -0.4042],
                                        [ 1.5550, -0.0704,  0.4447, -0.4042],
                                        [ 0.0302, -0.1819,  0.0508,  0.4408],
                                        [ 0.0302, -0.1819,  0.0508,  0.4408],
                                        [ 0.0302, -0.1819,  0.0508,  0.4408],
                                        [ 0.0302, -0.1819,  0.0508,  0.4408]]]
            parent_blocks_mask: [1, 6, 4] = [[[1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.]]]
            granularity_mask: [1, 6, 3, 4] = [[[[1., 1., 1., 1.], [1., 1., 0., 0.], [0., 0., 0., 0.]],
                        [[1., 1., 1., 1.], [0., 0., 1., 1.], [0., 0., 0., 0.]],
                        [[0., 0., 0., 0.], [1., 1., 0., 0.], [1., 0., 0., 0.]],
                        [[0., 0., 0., 0.], [1., 1., 0., 0.], [0., 1., 0., 0.]],
                        [[0., 0., 0., 0.], [0., 0., 1., 1.], [0., 0., 1., 0.]],
                        [[0., 0., 0., 0.], [0., 0., 1., 1.], [0., 0., 0., 1.]]]]
        """
        batch, patch_num, patch_len = x.shape
        weights, indices = self.moe(x)
        expert_indices = indices.view(batch, patch_num, -1)  # [batch, patch_num, n_experts]
        weights = weights.view(batch, patch_num, -1)  # [batch, patch_num, n_experts]
        n_real_experts = self.moe.n_real_experts
        is_real_expert_mask = (expert_indices < n_real_experts)
        masked_indices = torch.where(is_real_expert_mask, expert_indices, -1)
        indices, _ = torch.max(masked_indices, dim=-1)  # [batch, patch_num]
        indices[indices == -1] = 0

        # Create initial parent mapping and blocks
        original_patches, original_mask, parent_mapping = _create_initial_setup(x, mask, size)

        # Save original expert indices for granularity mask generation
        original_expert_indices = expert_indices

        # Initialize position mapping for tracking patch positions within original patches
        position_mapping = _create_initial_position_mapping(x, size)

        # Apply division to both x and mask
        for i in range(self.levels - 1):
            current_patch_num = x.size(1)
            to_divide = (indices > i).to(x.device)
            div_counts = to_divide.sum(dim=1)  # [B]

            new_x, new_size, weights, expert_indices = _divide_patches(x, size, to_divide, weights, expert_indices)
            new_mask, _ = _divide_patches(mask, size, to_divide)

            # Update parent mapping and position mapping
            new_parent_mapping = _update_parent_mapping(parent_mapping, to_divide, div_counts, x.device)
            new_position_mapping = _update_position_mapping(position_mapping, to_divide, div_counts, x.device)

            new_patch_nums = current_patch_num + div_counts
            total_elements = new_patch_nums.sum().item()

            expand_mask = torch.stack([
                torch.ones_like(to_divide),
                to_divide
            ], dim=-1).view(batch, -1)  # [B, 2L]

            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, 2).reshape(batch, -1)
            valid_indices = expanded_indices[expand_mask]

            assert expand_mask.sum().item() == total_elements, \
                f"Mask sum {expand_mask.sum().item()} vs total {total_elements}"
            assert valid_indices.numel() == total_elements, \
                f"Indices {valid_indices.numel()} vs total {total_elements}"

            max_len = new_patch_nums.max()
            indices = torch.full((batch, max_len), -1, dtype=torch.long, device=x.device)

            valid_pos_mask = (torch.arange(max_len, device=x.device)[None, :] < new_patch_nums[:, None])

            indices[valid_pos_mask] = valid_indices

            x, mask, size = new_x, new_mask, new_size
            parent_mapping = new_parent_mapping
            position_mapping = new_position_mapping

        parent_blocks = _map_to_parent_blocks(original_patches, parent_mapping, x.shape)
        parent_blocks_mask = _map_to_parent_blocks(original_mask, parent_mapping, mask.shape)
        granularity_mask = _create_granularity_mask(original_expert_indices, parent_mapping, position_mapping,
                                                    x.shape, patch_len)
        # Generate x_final with rearranged features
        x_final = _generate_x_final(parent_blocks, parent_blocks_mask, granularity_mask)

        return new_x, new_mask, new_size, weights, expert_indices, x_final


class KairosModel(T5PreTrainedModel):
    config_class = KairosConfig
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: KairosConfig):
        super().__init__(config)
        self.model_dim = config.d_model
        self.config = config
        self.num_segments = getattr(self.config, "num_decoder_segments", 1)
        if self.config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack_rope(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.config.quantiles)
        quantiles = torch.tensor(self.config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
            position_embedding_type=self.config.position_embedding_type
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # FFT feature normalization
        self.fft_norm = nn.LayerNorm(self.config.instance_rope_input_feature_dim)

        self.patch = DynamicPatch(
            max_patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
            levels=config.levels,
            n_null_experts=config.n_null_experts,
            n_activated_experts=config.n_activated_experts,
            moe_inter_dim=config.moe_inter_dim,
            update_bias_rate=config.update_bias_rate,
            target_dist=config.target_dist,
            route_scale=config.route_scale,
        )
        in_dim_ls = [config.input_patch_size]
        current_patch_size = config.input_patch_size
        for _ in range(config.levels - 1):
            assert current_patch_size % 2 == 0
            current_patch_size = current_patch_size // 2
            in_dim_ls.append(current_patch_size)
        self.input_patch_embedding = MultiInResidualBlock(
            in_dim_ls=tuple(in_dim_ls),
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
            position_embedding_type=self.config.position_embedding_type
        )
        self.loss_weight_scheme = config.loss_weight_scheme

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()
        elif isinstance(module, MultiInResidualBlock):
            module.hidden_layer.reset_parameters()
            module.residual_layer.reset_parameters()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack_rope(decoder_config, self.shared)

    def fft_process(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Process the input context with FFT, masked appropriately.

        Args:
            context: Input time series data, shape (batch_size, seq_length)
            mask: Mask indicating valid values, shape (batch_size, seq_length)

        Returns:
            Normalized FFT features, shape (batch_size, self.config.s_feature)
        """
        masked_context = torch.where(mask.bool(), context, torch.zeros_like(context))
        fft_result = torch.fft.rfft(masked_context, dim=-1)
        fft_amplitude = torch.abs(fft_result)

        s_feature = self.config.instance_rope_input_feature_dim
        fft_length = fft_amplitude.shape[-1]

        if fft_length >= s_feature:
            fft_features = fft_amplitude[..., :s_feature]
        else:
            padding = torch.zeros((fft_amplitude.shape[0], s_feature - fft_length),
                                  device=fft_amplitude.device, dtype=fft_amplitude.dtype)
            fft_features = torch.cat([fft_amplitude, padding], dim=-1)

        fft_normalized = self.fft_norm(fft_features)

        return fft_normalized

    def encode(
            self, context: torch.Tensor, mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape
        if context.shape[-1] > self.config.context_length:
            context = context[..., -self.config.context_length:]
            mask = mask[..., -self.config.context_length:]

        # scaling
        context = torch.where(mask > 0.0, context, torch.nan)
        context, loc_scale = self.instance_norm(context)
        s_features = self.fft_process(context, mask)
        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context, patched_mask, size, expert_weights, expert_indices, x_final = self.patch(context, mask)
        patched_mask = torch.nan_to_num(patched_mask, nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context, size, expert_weights, expert_indices, x_final)

        if self.config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            return_dict=True,
            output_attentions=output_attentions,
            s_features=s_features,
            size=size
        )

        return encoder_outputs.last_hidden_state, loc_scale, input_embeds, attention_mask, encoder_outputs.attentions, s_features

    def forward(
            self,
            past_target: torch.Tensor,
            past_is_pad: Optional[torch.Tensor] = None,
            past_observed_values: Optional[torch.Tensor] = None,
            future_target: Optional[torch.Tensor] = None,
            future_is_pad: Optional[torch.Tensor] = None,
            future_observed_values: Optional[torch.Tensor] = None,
            generation: bool = False,
            prediction_length: Optional[int] = None,
            output_attentions: bool = False,
            infer_is_positive: Optional[bool] = False,
            force_flip_invariance: Optional[bool] = False,
            *args,
            **kwargs,
    ) -> KairosOutput:
        if generation:
            return self.generate(
                past_target=past_target,
                past_is_pad=past_is_pad,
                past_observed_values=past_observed_values,
                prediction_length=prediction_length,
                infer_is_positive=infer_is_positive,
                force_flip_invariance=force_flip_invariance,
            )
        batch_size = past_target.size(0)
        # Generate masks based on padding and observed values
        mask = ~torch.isnan(past_target)
        if past_is_pad is not None:
            mask = mask & ~past_is_pad.bool()

        if past_observed_values is not None:
            mask = mask & past_observed_values.bool()

        future_mask = ~torch.isnan(future_target) if future_target is not None else None
        if future_is_pad is not None and future_mask is not None:
            future_mask = future_mask & ~future_is_pad.bool()

        if future_observed_values is not None and future_mask is not None:
            future_mask = future_mask & future_observed_values.bool()

        (
            hidden_states,
            loc_scale,
            input_embeds,
            attention_mask,
            attention_scores,
            s_features
        ) = self.encode(context=past_target, mask=mask, output_attentions=output_attentions)

        sequence_output, cross_attention_scores = self.decode(
            input_embeds, attention_mask, hidden_states, output_attentions=output_attentions, s_features=s_features
        )

        total_prediction_length = self.num_segments * self.config.prediction_length
        # [B, N, num_quantiles * prediction_length]
        raw_preds = self.output_patch_embedding(sequence_output)

        # [B, N, Q * L] -> [B, N, Q, L]
        raw_preds_reshaped = raw_preds.view(
            batch_size,
            self.num_segments,
            self.num_quantiles,
            self.config.prediction_length
        )

        # [B, N, Q, L] -> [B, Q, N, L]
        raw_preds_permuted = raw_preds_reshaped.permute(0, 2, 1, 3)

        # [B, Q, N*L]
        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            total_prediction_length,
        )
        quantile_preds = raw_preds_permuted.contiguous().view(*quantile_preds_shape)

        loss = None
        if future_target is not None:
            # normalize target
            future_target, _ = self.instance_norm(future_target, loc_scale)
            future_target = future_target.unsqueeze(1)  # type: ignore
            assert self.num_segments * self.config.prediction_length >= future_target.shape[-1]

            future_target = future_target.to(quantile_preds.device)
            future_mask = (
                future_mask.unsqueeze(1).to(quantile_preds.device) & ~torch.isnan(future_target)
                if future_mask is not None
                else ~torch.isnan(future_target)
            )
            future_target[~future_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if total_prediction_length > future_target.shape[-1]:
                padding_shape = (
                    *future_target.shape[:-1],
                    total_prediction_length - future_target.shape[-1],
                )
                future_target = torch.cat([future_target, torch.zeros(padding_shape).to(future_target)], dim=-1)
                future_mask = torch.cat([future_mask, torch.zeros(padding_shape).to(future_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (future_target - quantile_preds)
                    * ((future_target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * future_mask.float()
            )
            # quantile_preds: [B, Q, N*L]

            loss = loss.nanmean(dim=-2)  # Mean over quantile levels

            if self.config.loss_weight_scheme == 'log_decay':
                weights = get_log_decay_weights(total_prediction_length, device=loss.device)
                loss = loss * weights
            loss = loss.nansum(dim=-1)  # Sum over prediction horizon

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)
        assert not torch.isnan(quantile_preds).any(), (
            f"{past_target[torch.isnan(quantile_preds).any(dim=-1).any(dim=-1)]}"
        )
        return KairosOutput(
            loss=loss,
            prediction_outputs=quantile_preds,
            attentions=attention_scores if output_attentions else None,
            cross_attentions=cross_attention_scores if output_attentions else None,
            future_target=future_target,
        )

    def _autoregressive_generate(
            self,
            past_target: torch.Tensor,
            past_is_pad: Optional[torch.Tensor],
            past_observed_values: Optional[torch.Tensor],
            prediction_length: int,
    ) -> torch.Tensor:
        central_idx = torch.abs(self.quantiles.clone().detach() - 0.5).argmin()
        max_pred_len_per_step = self.num_segments * self.config.prediction_length
        output = self(
            past_target=past_target,
            past_is_pad=past_is_pad,
            past_observed_values=past_observed_values,
            prediction_length=min(prediction_length, max_pred_len_per_step),
        )
        predictions_list = [output.prediction_outputs]
        remaining = prediction_length - max_pred_len_per_step
        while remaining > 0:
            current_prediction_chunk = predictions_list[-1]

            past_target = torch.cat([past_target, current_prediction_chunk[:, central_idx]], dim=-1)

            if past_observed_values is not None:
                past_observed_values = torch.cat(
                    [past_observed_values, torch.ones_like(current_prediction_chunk[:, central_idx])], dim=-1
                )
            if past_is_pad is not None:
                past_is_pad = torch.cat(
                    [past_is_pad, torch.zeros_like(current_prediction_chunk[:, central_idx])], dim=-1
                )
            output = self(
                past_target=past_target,
                past_is_pad=past_is_pad,
                past_observed_values=past_observed_values,
                prediction_length=min(remaining, max_pred_len_per_step),
            )
            predictions_list.append(output.prediction_outputs)
            remaining -= max_pred_len_per_step

        prediction = torch.cat(predictions_list, dim=-1)

        return prediction[:, :, :prediction_length]

    def generate(
            self,
            past_target: torch.Tensor,
            past_is_pad: Optional[torch.Tensor] = None,
            past_observed_values: Optional[torch.Tensor] = None,
            prediction_length: Optional[int] = None,
            infer_is_positive: Optional[bool] = False,
            force_flip_invariance: Optional[bool] = False,
    ) -> KairosOutput:

        if prediction_length is None:
            prediction_length = self.num_segments * self.config.prediction_length
        max_supported_len = self.num_segments * self.config.prediction_length
        if prediction_length > max_supported_len and not self.training:
            warnings.warn(
                f"Prediction length {prediction_length} is greater than the model's prediction length {max_supported_len}. "
            )
        is_positive_mask = None
        if infer_is_positive:
            is_positive_mask = (~torch.any(past_target < 0, dim=1)).view(-1, 1, 1)
        if force_flip_invariance:
            pred_original = self._autoregressive_generate(
                past_target, past_is_pad, past_observed_values, prediction_length
            )
            pred_flipped = self._autoregressive_generate(
                -past_target, past_is_pad, past_observed_values, prediction_length
            )

            pred_flipped_corrected = -torch.flip(pred_flipped, dims=[1])
            prediction = (pred_original + pred_flipped_corrected) / 2
        else:
            prediction = self._autoregressive_generate(
                past_target, past_is_pad, past_observed_values, prediction_length
            )

        if is_positive_mask is not None:
            prediction = torch.where(
                is_positive_mask,
                torch.maximum(prediction, torch.tensor(0.0, device=prediction.device)),
                prediction,
            )
        return KairosOutput(
            prediction_outputs=prediction,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack_rope(decoder_config, self.shared)

    def decode(
            self,
            input_embeds,
            attention_mask,
            hidden_states,
            output_attentions=False,
            s_features: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        if self.config.diff_decoder_token_id:
            if self.num_segments == 1:
                custom_ids = [0]
            else:
                custom_ids = [0] + list(range(2, self.num_segments + 1))
            assert len(custom_ids) == self.num_segments
            input_sequence = torch.tensor(custom_ids, device=input_embeds.device, dtype=torch.long)
            decoder_input_ids = input_sequence.unsqueeze(0).expand(batch_size, -1)
        else:
            decoder_input_ids = torch.full(
                (batch_size, self.num_segments),
                self.config.decoder_start_token_id,
                device=input_embeds.device,
            )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
            s_features=s_features
        )

        return decoder_outputs.last_hidden_state, decoder_outputs.cross_attentions  # sequence_outputs, b x 1 x d_model