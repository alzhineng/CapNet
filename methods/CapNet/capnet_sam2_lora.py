

from .lpa_ablation_common import LPAComponentAblation


class CapNetSAM2LoRA(LPAComponentAblation):
    def __init__(self, **kwargs):
        super().__init__(use_adapter=False, use_lora=True, **kwargs)
