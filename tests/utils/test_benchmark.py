"""Tests for the model benchmark utility and metric table."""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from lofop.training.evaluator import DetectionMetrics  # noqa: E402
from lofop.utils import (  # noqa: E402
    benchmark_model,
    count_flops,
    render_csv,
    render_json,
    render_table,
    write_reports,
)


class TestFlops:
    def test_conv_flops_exact(self):
        # 1 conv: out 4x8x8, kernel 3x3, in 3 channels -> 2*4*64*9*3 MAC-FLOPs.
        model = nn.Conv2d(3, 4, 3, padding=1)
        assert count_flops(model, image_size=8) == 2 * 4 * 8 * 8 * 9 * 3

    def test_linear_counted(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 4 * 4, 5)

            def forward(self, x):
                return self.fc(x.flatten(1))

        assert count_flops(Tiny(), image_size=4) == 2 * 5 * 48


class TestReportAndTable:
    def make_report(self, accuracy=None):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.SiLU())
        return benchmark_model(model, "tiny", image_size=16, accuracy=accuracy)

    def test_structural_metrics_measured(self):
        report = self.make_report()
        assert report.parameters == sum(
            p.numel() for p in nn.Conv2d(3, 4, 3, padding=1).parameters()
        )
        assert report.flops > 0
        assert report.size_mb > 0
        assert report.cpu_fps and report.cpu_fps > 0

    def test_table_dashes_without_accuracy(self):
        table = render_table([self.make_report()])
        assert "| mAP@50 | - |" in table
        assert "Parameters" in table and "FLOPs" in table and "Model Size" in table

    def test_table_fills_accuracy_when_measured(self):
        metrics = DetectionMetrics(
            map50=0.5, map50_95=0.3, precision=0.7, recall=0.6, per_class_ap50={}
        )
        table = render_table([self.make_report(accuracy=metrics)])
        assert "| mAP@50 | 0.5000 |" in table
        assert "| Recall | 0.6000 |" in table

    def test_reference_columns_render_verbatim(self):
        table = render_table(
            [self.make_report()],
            reference_columns={"Reference (paper)": {"Parameters": "20,000,000"}},
        )
        assert "Reference (paper)" in table
        assert "20,000,000" in table


class TestCsvJsonExport:
    def make_report(self, accuracy=None):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.SiLU())
        return benchmark_model(model, "tiny", image_size=16, accuracy=accuracy)

    def test_record_is_flat_and_none_safe(self):
        record = self.make_report().to_record()
        assert record["name"] == "tiny" and record["parameters"] > 0
        assert record["map50"] is None  # no accuracy supplied

    def test_record_fills_accuracy(self):
        metrics = DetectionMetrics(
            map50=0.5, map50_95=0.3, precision=0.7, recall=0.6, per_class_ap50={}, f1=0.65
        )
        record = self.make_report(accuracy=metrics).to_record()
        assert record["map50"] == 0.5 and record["f1"] == 0.65

    def test_csv_has_header_and_row(self):
        csv_text = render_csv([self.make_report()])
        lines = csv_text.strip().splitlines()
        assert lines[0].startswith("name,parameters,flops")
        assert len(lines) == 2

    def test_json_roundtrips(self):
        import json

        data = json.loads(render_json([self.make_report()]))
        assert isinstance(data, list) and data[0]["name"] == "tiny"

    def test_write_reports_creates_three_files(self, tmp_path):
        written = write_reports([self.make_report()], tmp_path / "res")
        assert set(written) == {"md", "csv", "json"}
        assert all(path.is_file() for path in written.values())
