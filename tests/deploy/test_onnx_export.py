"""Tests for ONNX export, verification, and torch-free post-processing."""

import pytest

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")

from lofop.deploy import DenseExportWrapper, export_onnx, postprocess_dense  # noqa: E402
from lofop.models import ApexHead, DeltaFusion, LofopDetect, RidgeNet  # noqa: E402

IMAGE_SIZE = 64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    backbone = RidgeNet(widths=(16, 32, 64, 128), depths=(1, 1, 1, 1))
    neck = DeltaFusion(in_channels=backbone.out_channels, width=32)
    head = ApexHead(num_classes=3, width=32, num_convs=1)
    return LofopDetect(backbone, neck, head).eval()


@pytest.fixture(scope="module")
def exported(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model, path, image_size=IMAGE_SIZE, verify=True)
    return path


class TestExport:
    def test_file_written_and_verified(self, exported):
        assert exported.is_file() and exported.stat().st_size > 1000

    def test_wrapper_output_shapes(self, model):
        boxes, scores = DenseExportWrapper(model)(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        locations = (8 * 8) + (4 * 4) + (2 * 2)  # 64px input over strides 8/16/32
        assert boxes.shape == (1, locations, 4)
        assert scores.shape == (1, locations, 3)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_ort_matches_torch(self, model, exported):
        example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.inference_mode():
            torch_boxes, torch_scores = DenseExportWrapper(model)(example)
        session = onnxruntime.InferenceSession(
            str(exported), providers=["CPUExecutionProvider"]
        )
        ort_boxes, ort_scores = session.run(None, {"images": example.numpy()})
        assert float((torch_boxes - torch.from_numpy(ort_boxes)).abs().max()) < 1e-4
        assert float((torch_scores - torch.from_numpy(ort_scores)).abs().max()) < 1e-4

    def test_verification_catches_divergence(self, model, tmp_path):
        from lofop.core.exceptions import LofopError
        from lofop.deploy.onnx_export import verify_onnx

        path = tmp_path / "model.onnx"
        export_onnx(model, path, image_size=IMAGE_SIZE, verify=False)
        with pytest.raises(LofopError, match="diverge"):
            verify_onnx(
                DenseExportWrapper(model), path,
                torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE), tolerance=-1.0,
            )

    def test_dynamic_export_runs_at_unseen_sizes(self, model, tmp_path):
        # Dynamic export is verified internally at two sizes; here we also run
        # a third, unseen resolution and assert the point count tracks it.
        path = tmp_path / "dynamic.onnx"
        export_onnx(model, path, image_size=IMAGE_SIZE, dynamic=True, verify=True)
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        small, _ = session.run(None, {"images": torch.randn(1, 3, 96, 96).numpy()})
        large, _ = session.run(None, {"images": torch.randn(1, 3, 128, 160).numpy()})
        assert small.shape[1] < large.shape[1]  # more locations at higher resolution
        assert small.shape[2] == 4 and large.shape[2] == 4


class TestPostprocess:
    def test_thresholds_and_nms(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [80, 80, 90, 90]]
        scores = [
            [0.9, 0.0], [0.8, 0.0],   # same class, overlapping: second suppressed
            [0.0, 0.7],               # other class, disjoint: kept
            [0.1, 0.1],               # below threshold: dropped
        ]
        result = postprocess_dense(boxes, scores, score_threshold=0.25, nms_iou=0.5)
        assert len(result) == 2
        assert result.labels == [0, 1]
        assert result.scores == [0.9, 0.7]

    def test_empty_input(self):
        result = postprocess_dense([], [], score_threshold=0.25)
        assert len(result) == 0

    def test_end_to_end_with_ort(self, exported):
        session = onnxruntime.InferenceSession(
            str(exported), providers=["CPUExecutionProvider"]
        )
        example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).numpy()
        ort_boxes, ort_scores = session.run(None, {"images": example})
        result = postprocess_dense(
            ort_boxes[0].tolist(), ort_scores[0].tolist(), score_threshold=0.01
        )
        assert len(result) <= 300
        assert all(len(box) == 4 for box in result.boxes)
