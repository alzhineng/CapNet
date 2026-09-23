import abc
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SAM2 import SAM2
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from .zoomnext import dda_loss

LOGGER = logging.getLogger("main")


class _CapNetBase(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        if iter_percentage < min_point:
            return 0
        if iter_percentage > max_point:
            return 1
        if method == "linear":
            return (iter_percentage - min_point) / (max_point - min_point)
        perc = (iter_percentage - min_point) / (max_point - min_point)
        return (1 - math.cos(perc * math.pi)) / 2

    @abc.abstractmethod
    def body(self, data):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)
        if not self.training:
            return logits

        mask = data["mask"]
        prob = logits.sigmoid()
        sod_loss = dda_loss(logits, mask)
        losses = [sod_loss]
        loss_str = [f"bce: {sod_loss.item():.5f}"]

        if "depth" in data:
            depth = data["depth"]
            depth = depth - depth.amin(dim=(-2, -1), keepdim=True)
            depth = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * ((1 - (2 * prob - 1).abs().pow(2)) + depth).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

        return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if "shared_lora" in name or "prompt_learn" in name or "gate" in name:
                param.requires_grad = True
                param_groups["pretrained"].append(param)
            elif name.startswith("encoder.encoder."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            else:
                param.requires_grad = True
                param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class LinkedProxyAdapter(nn.Module):
    def __init__(self, channels, hidden_ratio=0.25):
        super().__init__()
        hidden = max(16, int(channels * hidden_ratio))
        self.high = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.lora_down = nn.Conv2d(hidden, 4, 1, bias=False)
        self.lora_up = nn.Conv2d(4, hidden, 1, bias=False)
        self.low = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        nn.init.zeros_(self.lora_up.weight)
        nn.init.zeros_(self.low[-1].weight)
        nn.init.zeros_(self.low[-1].bias)

    def forward(self, x):
        z = self.high(x)
        delta_w = self.lora_up(self.lora_down(z))
        delta_b = self.low(z + delta_w)
        return x + self.gate(x) * F.gelu(delta_b)


class ProgressiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.out = ConvBNReLU(channels, channels, 1)

    def _align(self, feat, size):
        if feat.shape[-2:] == size:
            return feat
        if feat.shape[-2] > size[0] or feat.shape[-1] > size[1]:
            return resize_to(feat, tgt_hw=size)
        return F.adaptive_max_pool2d(feat, size) + F.adaptive_avg_pool2d(feat, size)

    def _phi(self, a, b):
        b = self._align(b, a.shape[-2:])
        bsz, channels, height, width = a.shape
        seq_a = a.flatten(2)
        seq_b = b.flatten(2).transpose(1, 2)
        context = self.refine(seq_a).view(bsz, channels, height, width)
        gate = self.gate(seq_b).transpose(1, 2).view(bsz, channels, height, width)
        return self.out(context * gate)

    def forward(self, rgb, modalities):
        fused = rgb
        for feat in modalities:
            fused = self._phi(fused, feat)
        return fused


class HaarHighFrequency(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.project = ConvBNReLU(3, out_channels, 3, 1, 1)

    def forward(self, x, size):
        gray = x.mean(dim=1, keepdim=True)
        if gray.shape[-2] % 2 == 1:
            gray = gray[..., :-1, :]
        if gray.shape[-1] % 2 == 1:
            gray = gray[..., :, :-1]
        x00 = gray[..., 0::2, 0::2]
        x01 = gray[..., 0::2, 1::2]
        x10 = gray[..., 1::2, 0::2]
        x11 = gray[..., 1::2, 1::2]
        lh = x00 - x01 + x10 - x11
        hl = x00 + x01 - x10 - x11
        hh = x00 - x01 - x10 + x11
        high = torch.cat([lh.abs(), hl.abs(), hh.abs()], dim=1)
        high = resize_to(high, tgt_hw=size)
        return torch.softmax(self.project(high), dim=1)


class FrequencyGuidedDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.embed = ConvBNReLU(channels, channels, 1)
        self.dilated = nn.ModuleList(
            [ConvBNReLU(channels, channels, 3, 1, dilation, dilation=dilation) for dilation in (1, 3, 5, 7)]
        )
        self.scale = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.Softmax(dim=1))
        self.wavelet = HaarHighFrequency(channels)
        self.predict = nn.Sequential(
            ConvBNReLU(channels, channels // 2, 3, 1, 1),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, feat, rgb, out_size):
        emb = self.embed(feat)
        weights = self.scale(emb)
        multi = sum(branch(emb) * weights[:, idx : idx + 1] for idx, branch in enumerate(self.dilated))
        guided = multi * self.wavelet(rgb, multi.shape[-2:])
        return resize_to(self.predict(guided), tgt_hw=out_size)


class CapNet(_CapNetBase):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
        num_zoom=3,
        num_split=2,
    ):
        super().__init__()
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.encoder = SAM2()
        self.embed_dims = [144, 288, 576, 1152]

        self.adapters = nn.ModuleList([LinkedProxyAdapter(dim) for dim in self.embed_dims])
        self.transforms = nn.ModuleList([ConvBNReLU(dim, mid_dim, 1) for dim in self.embed_dims])
        self.fusions = nn.ModuleList([ProgressiveFusion(mid_dim) for _ in self.embed_dims])
        self.top_down = nn.ModuleList([ConvBNReLU(mid_dim, mid_dim, 3, 1, 1) for _ in range(3)])
        self.decoder = FrequencyGuidedDecoder(mid_dim)

    def _encode(self, x):
        x = self.normalizer(x)
        feats = self.encoder(x)
        return [trans(adapter(feat)) for feat, adapter, trans in zip(feats, self.adapters, self.transforms)]

    def _modalities(self, data):
        return list(data["imgs"])

    def body(self, data):
        imgs = self._modalities(data)
        out_size = imgs[0].shape[-2:]

        encoded = [self._encode(img) for img in imgs]
        fused = []
        for level in range(len(self.embed_dims)):
            rgb = encoded[0][level]
            aux = [modal[level] for modal in encoded[1:]]
            fused.append(self.fusions[level](rgb, aux))

        x = fused[-1]
        for idx, level in enumerate((2, 1, 0)):
            x = self.top_down[idx](resize_to(x, tgt_hw=fused[level].shape[-2:]) + fused[level])
        return self.decoder(x, imgs[0], out_size)
