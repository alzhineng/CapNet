"""Experiment 2: frozen SAM2 + rank-4 LoRA."""

from .lpa_ablation_common import LPAComponentAblation


class CapNetSAM2LoRA(LPAComponentAblation):
    def __init__(self, **kwargs):
        super().__init__(use_adapter=False, use_lora=True, **kwargs)
