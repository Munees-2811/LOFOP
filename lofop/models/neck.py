"""DeltaFusion: LOFOP's bidirectional feature fusion neck.

Lateral 1x1 projections bring C3/C4/C5 to one width; the stride-32 map gets
global :class:`PulseAttention`; a gated top-down pass then a gated bottom-up
pass produce P3/P4/P5. Fusion gates are learnable scalars normalized by
softmax, so training decides per level how much each direction contributes.
Design rationale: docs/lofop-detect.md section 2.2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.blocks import ConvNormAct, PulseAttention
from lofop.registries import NECKS


class _GatedFuse(nn.Module):
    """Fuse two same-shape maps with softmax-normalized learnable weights."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2))
        self.refine = ConvNormAct(channels, channels)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        gate = torch.softmax(self.weights, dim=0)
        return self.refine(gate[0] * a + gate[1] * b)


@NECKS.register()
class DeltaFusion(nn.Module):
    """Bidirectional pyramid fusion with attention on the coarsest level.

    Args:
        in_channels: Channels of the incoming C3/C4/C5 maps.
        width: Common channel width of the outgoing P3/P4/P5 maps.

    Forward takes ``[C3, C4, C5]`` and returns ``[P3, P4, P5]``, all with
    ``width`` channels at strides 8/16/32.
    """

    def __init__(self, in_channels: tuple[int, int, int], width: int = 96) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ModelError(
                "DeltaFusion expects exactly 3 input levels",
                context={"in_channels": list(in_channels)},
            )
        self.width = width
        self.laterals = nn.ModuleList(
            ConvNormAct(c, width, kernel_size=1) for c in in_channels
        )
        self.context = PulseAttention(width)
        self.top_down = nn.ModuleList(_GatedFuse(width) for _ in range(2))
        self.downsamples = nn.ModuleList(ConvNormAct(width, width, stride=2) for _ in range(2))
        self.bottom_up = nn.ModuleList(_GatedFuse(width) for _ in range(2))

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 3:
            raise ModelError(
                "DeltaFusion received the wrong number of feature maps",
                context={"got": len(features)},
            )
        c3, c4, c5 = (lat(f) for lat, f in zip(self.laterals, features))
        c5 = self.context(c5)

        m4 = self.top_down[0](c4, F.interpolate(c5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.top_down[1](c3, F.interpolate(m4, size=c3.shape[-2:], mode="nearest"))

        p4 = self.bottom_up[0](m4, self.downsamples[0](p3))
        p5 = self.bottom_up[1](c5, self.downsamples[1](p4))
        return [p3, p4, p5]
