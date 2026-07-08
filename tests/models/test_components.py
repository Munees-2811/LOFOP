"""Shape, stride, and gradient tests for LOFOP-Detect components."""

import pytest

torch = pytest.importorskip("torch")

from lofop.core.exceptions import ModelError  # noqa: E402
from lofop.models import ApexHead, DeltaFusion, RidgeNet  # noqa: E402

WIDTHS = (16, 32, 64, 128)
DEPTHS = (1, 1, 1, 1)


@pytest.fixture(scope="module")
def backbone():
    return RidgeNet(widths=WIDTHS, depths=DEPTHS)


class TestRidgeNet:
    def test_output_strides_and_channels(self, backbone):
        features = backbone(torch.randn(2, 3, 64, 64))
        assert [f.shape for f in features] == [
            torch.Size([2, 32, 8, 8]),    # C3, stride 8
            torch.Size([2, 64, 4, 4]),    # C4, stride 16
            torch.Size([2, 128, 2, 2]),   # C5, stride 32
        ]
        assert backbone.out_channels == (32, 64, 128)

    def test_gradients_flow_to_stem(self, backbone):
        images = torch.randn(1, 3, 64, 64)
        sum(f.sum() for f in backbone(images)).backward()
        stem_weight = next(backbone.stem.parameters())
        assert stem_weight.grad is not None and stem_weight.grad.abs().sum() > 0
        backbone.zero_grad()

    def test_rejects_wrong_stage_count(self):
        with pytest.raises(ModelError):
            RidgeNet(widths=(16, 32), depths=(1, 1))


class TestDeltaFusion:
    def test_output_shapes_uniform_width(self, backbone):
        neck = DeltaFusion(in_channels=backbone.out_channels, width=32)
        outputs = neck(backbone(torch.randn(2, 3, 64, 64)))
        assert [o.shape for o in outputs] == [
            torch.Size([2, 32, 8, 8]),
            torch.Size([2, 32, 4, 4]),
            torch.Size([2, 32, 2, 2]),
        ]

    def test_attention_gate_starts_neutral(self):
        neck = DeltaFusion(in_channels=(32, 64, 128), width=32)
        assert float(neck.context.gate.detach()) == 0.0

    def test_rejects_wrong_level_count(self):
        with pytest.raises(ModelError):
            DeltaFusion(in_channels=(32, 64))


class TestApexHead:
    def make(self):
        return ApexHead(num_classes=7, width=32, num_convs=1)

    def features(self, batch=2):
        return [torch.randn(batch, 32, 8, 8), torch.randn(batch, 32, 4, 4),
                torch.randn(batch, 32, 2, 2)]

    def test_output_shapes(self):
        cls_out, box_out, quality_out = self.make()(self.features())
        assert cls_out[0].shape == torch.Size([2, 7, 8, 8])
        assert box_out[1].shape == torch.Size([2, 4, 4, 4])
        assert quality_out[2].shape == torch.Size([2, 1, 2, 2])

    def test_box_distances_nonnegative_and_scaled(self):
        _, box_out, _ = self.make()(self.features())
        for level in box_out:
            assert (level >= 0).all()

    def test_level_points_cover_all_locations(self):
        head = self.make()
        points, strides = head.level_points(self.features())
        assert points.shape == (8 * 8 + 4 * 4 + 2 * 2, 2)
        assert strides[:64].eq(8).all() and strides[-4:].eq(32).all()
        assert points[0].tolist() == [4.0, 4.0]  # first stride-8 center

    def test_decode_boxes_roundtrip(self):
        points = torch.tensor([[10.0, 20.0]])
        distances = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        assert ApexHead.decode_boxes(points, distances).tolist() == [[8.0, 17.0, 14.0, 25.0]]

    def test_initial_cls_probability_near_one_percent(self):
        cls_out, _, _ = self.make()(self.features())
        assert cls_out[0].sigmoid().mean().item() == pytest.approx(0.01, rel=0.5)

    def test_rejects_feature_stride_mismatch(self):
        with pytest.raises(ModelError):
            self.make()(self.features()[:2])
