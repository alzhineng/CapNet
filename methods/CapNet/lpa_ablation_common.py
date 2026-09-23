
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.build_sam import build_sam2

from .capnet import _CapNetBase
from .ops import ConvBNReLU, PixelNormalizer, resize_to


LOGGER = logging.getLogger("main")


class SharedLoRA(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super().__init__()
        self.proj_down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.proj_up = nn.Linear(hidden_dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.zeros_(self.proj_up.weight)
        nn.init.zeros_(self.proj_up.bias)

    def forward(self, x):
        delta = self.proj_up(self.act(self.proj_down(x)))
        return self.gate(x) * delta


class AdaptedSAM2Block(nn.Module):
    def __init__(self, block, lora_by_dim, use_adapter, use_lora):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.adapter = BottleneckAdapter(dim) if use_adapter else None
        self.lora = lora_by_dim[str(dim)] if use_lora else None

    def forward(self, x):
        block_input = x
        if self.adapter is not None:
            block_input = block_input + self.adapter(block_input)
        output = self.block(block_input)
        if self.lora is not None and output.shape[-1] == x.shape[-1]:
            output = output + self.lora(x)
        return output


class AblationSAM2Encoder(nn.Module):
    def __init__(self, use_adapter, use_lora):
        super().__init__()
        checkpoint = Path(__file__).resolve().parents[2] / "sam2_hiera_large.pt"
        model = build_sam2("sam2_hiera_l.yaml", str(checkpoint))
        self.trunk = model.image_encoder.trunk

        for parameter in self.trunk.parameters():
            parameter.requires_grad = False

        dims = sorted({block.attn.qkv.in_features for block in self.trunk.blocks})
        self.shared_lora = nn.ModuleDict(
            {str(dim): SharedLoRA(dim=dim, rank=4) for dim in dims}
        ) if use_lora else nn.ModuleDict()
        self.trunk.blocks = nn.Sequential(
            *[
                AdaptedSAM2Block(
                    block=block,
                    lora_by_dim=self.shared_lora,
                    use_adapter=use_adapter,
                    use_lora=use_lora,
                )
                for block in self.trunk.blocks
            ]
        )

    def forward(self, x):
        return self.trunk(x)


class PlainPyramidDecoder(nn.Module):
    """A shared non-FGD decoder used identically by all LPA ablations."""

    def __init__(self, in_dims=(144, 288, 576, 1152), channels=64):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvBNReLU(in_dim, channels, 1) for in_dim in in_dims]
        )
        self.refine = nn.ModuleList(
            [ConvBNReLU(channels, channels, 3, 1, 1) for _ in range(3)]
        )
        self.predict = nn.Sequential(
            ConvBNReLU(channels, channels // 2, 3, 1, 1),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, features, out_size):
        features = [projection(feature) for projection, feature in zip(self.projections, features)]
        x = features[-1]
        for block, skip in zip(self.refine, reversed(features[:-1])):
            x = block(resize_to(x, tgt_hw=skip.shape[-2:]) + skip)
        return resize_to(self.predict(x), tgt_hw=out_size)


class LPAComponentAblation(_CapNetBase):
    def __init__(
        self,
        use_adapter,
        use_lora,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        **kwargs,
    ):
        super().__init__()
        del pretrained, num_frames, kwargs
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = AblationSAM2Encoder(use_adapter=use_adapter, use_lora=use_lora)
        self.decoder = PlainPyramidDecoder()

    def body(self, data):
        # LPA ablations use RGB only. Auxiliary modalities are intentionally ignored.
        rgb = data["imgs"][0]
        features = self.encoder(self.normalizer(rgb))
        return self.decoder(features, out_size=rgb.shape[-2:])

    def get_grouped_params(self):
        groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, parameter in self.named_parameters():
            is_adaptation_parameter = (
                ".adapter." in name
                or ".lora." in name
                or name.startswith("encoder.shared_lora.")
            )
            if name.startswith("encoder.trunk") and not is_adaptation_parameter:
                parameter.requires_grad = False
                groups["fixed"].append(parameter)
            else:
                parameter.requires_grad = True
                groups["retrained"].append(parameter)
        LOGGER.info(
            "Ablation parameter groups: fixed=%d, retrained=%d",
            len(groups["fixed"]),
            len(groups["retrained"]),
        )
        return groups
