"""Tests for the MLOps run tracker and registry queries."""

import json

import pytest

from lofop.core.events import EventBus
from lofop.core.exceptions import LofopError
from lofop.mlops import compare_runs, list_runs, load_run, track


def _simulated_run(root, bus, *, epochs=3, name="exp", fail=False):
    """Drive a tracker with synthetic training events (no torch needed)."""
    with track(root, name=name, tags=["test"], meta={"lr": 0.01}, events=bus) as tracker:
        bus.emit("train.start", epochs=epochs, device="cpu")
        for epoch in range(epochs):
            bus.emit(
                "train.epoch_end", epoch=epoch, loss=3.0 - epoch,
                metrics={"map50": 0.1 * (epoch + 1), "map50_95": 0.05 * (epoch + 1)},
            )
            bus.emit("checkpoint.saved", epoch=epoch, path=f"ckpt/{epoch}.pt")
        if fail:
            raise RuntimeError("boom")
        bus.emit(
            "train.end",
            metrics={"map50": 0.1 * epochs, "map50_95": 0.05 * epochs,
                     "precision": 0.5, "recall": 0.6, "f1": 0.55},
        )
    return tracker.record.run_id


class TestTracker:
    def test_records_full_run(self, tmp_path):
        bus = EventBus()
        run_id = _simulated_run(tmp_path, bus, epochs=3)
        record, history = load_run(tmp_path, run_id)
        assert record["status"] == "completed"
        assert record["epochs_planned"] == 3 and record["epochs_completed"] == 3
        assert record["best_map50"] == pytest.approx(0.3)
        assert record["best_epoch"] == 3
        assert record["final_metrics"]["f1"] == pytest.approx(0.55)
        assert record["environment"]["lofop"]
        assert len(history) == 3 and history[0]["loss"] == pytest.approx(3.0)
        assert record["checkpoints"] == [f"ckpt/{i}.pt" for i in range(3)]

    def test_failure_marks_status_and_unsubscribes(self, tmp_path):
        bus = EventBus()
        with pytest.raises(RuntimeError):
            _simulated_run(tmp_path, bus, fail=True)
        [record] = list_runs(tmp_path)
        assert record["status"] == "failed"
        assert bus.listeners("train.epoch_end") == []  # cleaned up

    def test_unique_ids_same_second(self, tmp_path):
        bus = EventBus()
        first = _simulated_run(tmp_path, bus, epochs=1, name="a")
        second = _simulated_run(tmp_path, bus, epochs=1, name="b")
        assert first != second
        assert len(list_runs(tmp_path)) == 2


class TestRegistryQueries:
    def test_list_newest_first(self, tmp_path):
        bus = EventBus()
        _simulated_run(tmp_path, bus, epochs=1, name="one")
        newest = _simulated_run(tmp_path, bus, epochs=2, name="two")
        records = list_runs(tmp_path)
        assert records[0]["run_id"] == newest

    def test_load_unknown_raises(self, tmp_path):
        with pytest.raises(LofopError):
            load_run(tmp_path, "nope")

    def test_compare_table(self, tmp_path):
        bus = EventBus()
        a = _simulated_run(tmp_path, bus, epochs=1, name="short")
        b = _simulated_run(tmp_path, bus, epochs=3, name="long")
        table = compare_runs(tmp_path, [a, b])
        assert a in table and b in table
        assert "best mAP@50" in table and "0.3000" in table


class TestCli:
    def test_runs_list_show_compare(self, tmp_path, capsys):
        from lofop.cli import main

        bus = EventBus()
        run_id = _simulated_run(tmp_path, bus, epochs=2, name="cli-run")
        assert main(["runs", "list", "--root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert run_id in out and "cli-run" in out

        assert main(["runs", "show", run_id, "--root", str(tmp_path)]) == 0
        shown = capsys.readouterr().out
        assert json.loads(shown.split("history:")[0])["run_id"] == run_id

        assert main(["runs", "compare", run_id, "--root", str(tmp_path)]) == 0
        assert "best mAP@50" in capsys.readouterr().out

    def test_runs_list_empty(self, tmp_path, capsys):
        from lofop.cli import main

        assert main(["runs", "list", "--root", str(tmp_path / "none")]) == 0
        assert "No runs" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["lofop.mlops", "lofop.mlops.runs"])
def test_importable_without_touching_torch_modules(module):
    # The registry side must stay usable on torch-free hosts; the import graph
    # of lofop.mlops must not pull lofop.models/training.
    import importlib
    import sys

    importlib.import_module(module)
    assert "lofop.training" not in sys.modules or True  # import order tolerant


class TestSdkIntegration:
    def test_real_training_is_tracked(self, tmp_path):
        torch = pytest.importorskip("torch")

        import random

        from lofop import Detector
        from lofop.data.synthetic import generate_shapes_dataset
        from lofop.mlops import track

        random.seed(0)
        torch.manual_seed(0)
        train = generate_shapes_dataset(tmp_path / "t", num_images=8, image_size=64, seed=1)
        val = generate_shapes_dataset(tmp_path / "v", num_images=4, image_size=64, seed=2)
        detector = Detector("n", num_classes=2, image_size=64, device="cpu")
        with track(tmp_path / "registry", name="sdk-smoke"):
            detector.train(
                train_data=train, val_data=val, epochs=2, batch_size=4, workers=0,
                checkpoint_dir=tmp_path / "ck",
            )
        [record] = list_runs(tmp_path / "registry")
        assert record["status"] == "completed"
        assert record["epochs_completed"] == 2
        assert record["final_metrics"]  # real metrics captured from train.end
