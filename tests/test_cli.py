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

    def test_framework_errors_exit_2(self, tmp_path, capsys):
        code = main([
            "dataset", "stats", "--format", "coco", "--source", str(tmp_path / "nope.json"),
        ])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_command_exits_with_usage(self):
        with pytest.raises(SystemExit):
            main(["dataset", "explode"])
