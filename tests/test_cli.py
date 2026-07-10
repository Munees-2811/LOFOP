"""Tests for the lofop CLI (invoked in-process through main())."""

import json

import pytest

from lofop.cli import main
from lofop.version import __version__


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


class TestDatasetCommands:
    def test_convert(self, coco_file, tmp_path, capsys):
        target = tmp_path / "yolo_out"
        code = main([
            "dataset", "convert", "--from", "coco", "--source", str(coco_file),
            "--to", "yolo", "--target", str(target),
        ])
        assert code == 0
        assert (target / "classes.txt").is_file()
        assert "coco -> yolo" in capsys.readouterr().out

    def test_validate_ok_dataset(self, yolo_root, capsys):
        code = main(["dataset", "validate", "--format", "yolo", "--source", str(yolo_root)])
        assert code == 0
        assert "OK" in capsys.readouterr().out

    def test_validate_bad_dataset_exits_nonzero(self, coco_file, tmp_path, capsys):
        bad = json.loads(coco_file.read_text())
        bad["annotations"][0]["bbox"] = [10, 10, 0, 0]   # degenerate
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps(bad))
        code = main([
            "dataset", "validate", "--format", "coco", "--source", str(bad_file),
            "--no-check-images",
        ])
        assert code == 1
        out = capsys.readouterr().out
        assert "degenerate" in out and "FAILED" in out

    def test_stats_markdown_and_output_file(self, coco_file, tmp_path, capsys):
        report = tmp_path / "stats.md"
        code = main([
            "dataset", "stats", "--format", "coco", "--source", str(coco_file),
            "-o", str(report),
        ])
        assert code == 0
        assert "| cat | 2 |" in capsys.readouterr().out
        assert report.is_file()

    def test_stats_json(self, coco_file, capsys):
        code = main(["dataset", "stats", "--format", "coco", "--source", str(coco_file), "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["num_images"] == 2

    def test_show_renders_images(self, coco_file, tmp_path, capsys):
        out_dir = tmp_path / "vis"
        code = main([
            "dataset", "show", "--format", "coco", "--source", str(coco_file),
            "--image-root", str(coco_file.parent / "imgs"), "-o", str(out_dir),
        ])
        assert code == 0
        assert "Rendered" in capsys.readouterr().out
        assert list(out_dir.glob("*.png"))

    def test_framework_errors_exit_2(self, tmp_path, capsys):
        code = main([
            "dataset", "stats", "--format", "coco", "--source", str(tmp_path / "nope.json"),
        ])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_command_exits_with_usage(self):
        with pytest.raises(SystemExit):
            main(["dataset", "explode"])


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "LOFOP" in out and __version__ in out
    assert "native ops:" in out


class TestModelCommands:
    """predict/evaluate need the models extra; skipped if torch is absent."""

    def _image(self, tmp_path):
        pytest.importorskip("torch")
        from PIL import Image

        path = tmp_path / "scene.png"
        Image.new("RGB", (128, 96), color=(40, 90, 160)).save(path)
        return path

    def test_predict_json(self, tmp_path, capsys):
        image = self._image(tmp_path)
        out_file = tmp_path / "preds.json"
        code = main([
            "predict", "--config", "n", "--num-classes", "3",
            "--source", str(image), "--size", "128", "--json", "-o", str(out_file),
        ])
        assert code == 0
        payload = json.loads(out_file.read_text())
        assert payload[0]["image"] == str(image)
        assert {"boxes", "scores", "labels"} <= payload[0].keys()

    def test_evaluate_json(self, coco_file, tmp_path, capsys):
        pytest.importorskip("torch")
        out_file = tmp_path / "metrics.json"
        code = main([
            "evaluate", "--config", "n", "--num-classes", "4",
            "--format", "coco", "--source", str(coco_file),
            "--image-root", str(coco_file.parent / "imgs"),
            "--size", "128", "--json", "-o", str(out_file),
        ])
        assert code == 0
        metrics = json.loads(out_file.read_text())
        assert {"map50", "f1", "confusion_matrix", "per_class_precision"} <= metrics.keys()
