"""RidgeNet: LOFOP's convolutional backbone.

A stem downsamples 4x, then four stages of :class:`RidgeBlock`s run at strides
4/8/16/32; the last three stage outputs (C3, C4, C5) feed the neck. Width and
depth per stage come from config, so the n/s/m model family is one
implementation. Design rationale: docs/lofop-detect.md section 2.1.
"""

from __future__ import annotations

from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.blocks import ConvNormAct, RidgeBlock
from lofop.registries import BACKBONES


@BACKBONES.register()
class RidgeNet(nn.Module):
    """Multi-scale feature extractor.

    Args:
        widths: Channel count of each of the four stages.
        depths: Number of RidgeBlocks in each stage.
        in_channels: Input image channels.
        expansion: Expansion ratio inside each RidgeBlock.

    Forward returns ``[C3, C4, C5]`` at strides 8/16/32 with channels
    ``widths[1:]``.
    """

    def __init__(
        self,
        widths: tuple[int, int, int, int] = (48, 96, 192, 384),
        depths: tuple[int, int, int, int] = (1, 2, 4, 2),
        in_channels: int = 3,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or len(depths) != 4:
            raise ModelError(
                "RidgeNet needs exactly 4 stage widths and depths",
                context={"widths": list(widths), "depths": list(depths)},
            )
        self.widths = tuple(widths)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0] // 2, stride=2),
            ConvNormAct(widths[0] // 2, widths[0], stride=2),
        )
        stages = []
        prev = widths[0]
        for stage_index, (width, depth) in enumerate(zip(widths, depths)):
            layers: list[nn.Module] = []
            if stage_index > 0:
                layers.append(ConvNormAct(prev, width, stride=2))
            elif prev != width:
                layers.append(ConvNormAct(prev, width, kernel_size=1))
            layers.extend(RidgeBlock(width, expansion) for _ in range(depth))
            stages.append(nn.Sequential(*layers))
            prev = width
        self.stages = nn.ModuleList(stages)

    @property
    def out_channels(self) -> tuple[int, int, int]:
        """Channels of the returned C3/C4/C5 maps."""
        return self.widths[1:]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[1:]
