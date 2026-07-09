"""Tests for the high-level Python SDK (lofop.sdk.Detector)."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from lofop import Detector  # noqa: E402  (lazy top-level export)
from lofop.core.exceptions import LofopError, ModelError  # noqa: E402
from lofop.data.synthetic import generate_shapes_dataset  # noqa: E402
from lofop.models import LofopDetect  # noqa: E402

TINY = {
    "model": {
        "type": "model/LofopDetect",
        "score_threshold": 0.01,
        "backbone": {"type": "backbone/RidgeNet",
                     "widths": [16, 32, 64, 128], "depths": [1, 1, 1, 1]},
        "neck": {"type": "neck/DeltaFusion", "in_channels": [32, 64, 128], "width": 32},
        "head": {"type": "head/ApexHead", "num_classes": 2, "width": 32, "num_convs": 1},
    }
}


def tiny_detector(**kwargs):
    return Detector(TINY, image_size=64, device="cpu", **kwargs)


class TestConstruction:
    def test_variant_names_resolve(self):
        for name in ("lofop-detect-n", "n"):
            det = Detector(name, num_classes=3, device="cpu")
            assert isinstance(det.model, LofopDetect)
            assert det.model.head.num_classes == 3

    def test_ex_variant_has_20m_parameters(self):
        det = Detector("ex", num_classes=80, device="cpu")
        assert 19_000_000 <= det.num_parameters <= 21_000_000

    def test_unknown_variant_lists_options(self):
        with pytest.raises(LofopError, match="lofop-detect-ex"):
            Detector("lofop-detect-zz", device="cpu")

    def test_dict_config_and_module_inputs(self):
        det = tiny_detector()
        rewrapped = Detector(det.model, image_size=64, device="cpu")
        assert rewrapped.model is det.model

    def test_class_names_validated_and_defaulted(self):
        det = tiny_detector(class_names=["cat", "dog"])
        assert det.class_names == ["cat", "dog"]
        assert tiny_detector().class_names == ["class_0", "class_1"]
        with pytest.raises(ModelError):
            tiny_detector(class_names=["only-one"])

    def test_repr_mentions_size(self):
        assert "parameters=" in repr(tiny_detector())


class TestPredict:
    def test_image_path_boxes_in_original_coordinates(self, tmp_path):
        # A 200x100 image: predict() must return boxes in 200x100 space,
        # clamped to the image, regardless of the 64px inference resolution.
        path = tmp_path / "img.png"
        Image.new("RGB", (200, 100), (30, 30, 30)).save(path)
        det = tiny_detector()
        (result,) = det.predict(path)
        for x1, y1, x2, y2 in result.boxes:
            assert 0 <= x1 <= 200 and 0 <= x2 <= 200
            assert 0 <= y1 <= 100 and 0 <= y2 <= 100
        assert len(result.boxes) == len(result.scores) == len(result.labels)

    def test_accepts_pil_tensor_and_batch_list(self, tmp_path):
        det = tiny_detector()
        pil = Image.new("RGB", (64, 64), (10, 10, 10))
        tensor = torch.rand(3, 64, 64)
        results = det.predict([pil, tensor])
        assert len(results) == 2

    def test_score_threshold_override_is_temporary(self):
        det = tiny_detector()
        before = det.model.score_threshold
        det.predict(torch.rand(3, 64, 64), score_threshold=0.9)
        assert det.model.score_threshold == before


class TestTrainEvaluateSaveExport:
    @pytest.fixture(scope="class")
    def shapes(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("sdk_shapes")
        return (
            generate_shapes_dataset(root / "train", num_images=12, image_size=64, seed=3),
            generate_shapes_dataset(root / "val", num_images=6, image_size=64, seed=4),
        )

    def test_train_evaluate_roundtrip(self, shapes, tmp_path):
        train, val = shapes
        det = tiny_detector()
        metrics = det.train(
            train_data=train, val_data=val, epochs=2, batch_size=4,
            lr=0.02, workers=0, checkpoint_dir=tmp_path / "run",
        )
        assert metrics is not None and 0.0 <= metrics.map50 <= 1.0
        again = det.evaluate(val)
        assert 0.0 <= again.map50 <= 1.0

    def test_train_requires_data(self):
        with pytest.raises(LofopError, match="train_data"):
            tiny_detector().train(epochs=1)

    def test_save_and_reload_weights(self, tmp_path):
        det = tiny_detector()
        path = det.save(tmp_path / "w.pt")
        reloaded = Detector(TINY, image_size=64, device="cpu", checkpoint=path)
        key = next(iter(det.model.state_dict()))
        assert torch.equal(det.model.state_dict()[key], reloaded.model.state_dict()[key])

    def test_export_onnx_by_suffix(self, tmp_path):
        pytest.importorskip("onnxruntime")
        det = tiny_detector()
        out = det.export(tmp_path / "m.onnx")
        assert out.is_file() and out.stat().st_size > 1000

    def test_export_unknown_format(self, tmp_path):
        with pytest.raises(LofopError, match="Unknown export format"):
            tiny_detector().export(tmp_path / "m.bin", format="bin")

    def test_optimize_chains(self):
        det = tiny_detector()
        assert det.optimize() is det and det.model._channels_last
