from transformers import AutoConfig, AutoModel

from .configuration_kairos import KairosConfig
from .modeling_kairos import KairosModel

AutoConfig.register("kairos", KairosConfig)
AutoModel.register(KairosConfig, KairosModel)

__all__ = ["KairosModel", "KairosConfig"]
