"""Building blocks shared by LOFOP model components.

Design rationale for each block lives in docs/lofop-detect.md; docstrings here
describe behavior and shapes only.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvNormAct(nn.Module):
    """Convolution + norm + activation, the basic composite unit.

    Uses BatchNorm (fusable into the conv for deployment) and SiLU by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, groups=groups, bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class RidgeBlock(nn.Module):
    """Residual inverted bottleneck: 7x7 depthwise -> expand -> project.

    Input and output shapes are identical; the residual connection is always
    active.
    """

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.spatial = ConvNormAct(channels, channels, kernel_size=7, groups=channels, act=False)
        self.expand = ConvNormAct(channels, hidden, kernel_size=1)
        self.project = ConvNormAct(hidden, channels, kernel_size=1, act=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.project(self.expand(self.spatial(x)))


class PulseAttention(nn.Module):
    """Single-head global self-attention over a small feature map.

    Intended for the stride-32 pyramid level only, where the token count is
    tiny (400 tokens at 640px input); cost grows quadratically with tokens.
    Residual, with a learnable output gate initialized to zero so inserting
    the block never degrades an untrained network.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(1))
        self.scale = channels ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        query, key, value = self.qkv(x).flatten(2).chunk(3, dim=1)  # (B, C, N)
        attn = torch.softmax(query.transpose(1, 2) @ key * self.scale, dim=-1)  # (B, N, N)
        out = (value @ attn.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.gate * self.proj(out)
