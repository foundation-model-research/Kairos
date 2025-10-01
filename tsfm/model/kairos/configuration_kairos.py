from typing import List

from transformers.models.t5.modeling_t5 import T5Config

class KairosConfig(T5Config):
    model_type = "kairos"

    def __init__(
        self,
        context_length: int = 2048,
        prediction_length: int = 64,
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        use_reg_token: bool = False,
        levels: int = 3,
        n_activated_experts: int = 1,
        moe_inter_dim: int = 1408,
        update_bias_rate: float = 0.001,
        target_dist: list = None,
        route_scale: float = 1.0,
        num_decoder_segments: int = 1,
        loss_weight_scheme: str = "none",
        n_null_experts: int = 0,
        # Instance Rope
        position_embedding_type: str = "instance_wise_rope",
        instance_rope_input_feature_dim: int = 128,
        min_period: str = "original_rope_init",
        max_period: str = "original_rope_init",
        rope_init: str = "exp",
        scale_method: str = "log",
        is_cross_attention_pe: bool = True,
        diff_decoder_token_id: bool = False,
        cross_attention_pe_flip: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_patch_size = input_patch_size
        self.input_patch_stride = input_patch_stride
        self.quantiles = quantiles
        self.use_reg_token = use_reg_token
        self.levels = levels
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.update_bias_rate = update_bias_rate
        self.target_dist = target_dist
        self.route_scale = route_scale
        self.num_decoder_segments = num_decoder_segments
        self.loss_weight_scheme = loss_weight_scheme
        self.n_null_experts = n_null_experts
        # Instance Rope
        self.position_embedding_type = position_embedding_type
        self.instance_rope_input_feature_dim = instance_rope_input_feature_dim
        self.min_period = min_period
        self.max_period = max_period
        self.rope_init = rope_init
        self.scale_method = scale_method
        self.is_cross_attention_pe = is_cross_attention_pe
        self.diff_decoder_token_id = diff_decoder_token_id
        self.cross_attention_pe_flip = cross_attention_pe_flip