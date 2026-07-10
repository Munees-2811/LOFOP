"""Config-driven schedulers, early stopping, and the TensorBoard hook."""

import pytest

torch = pytest.importorskip("torch")

from lofop.core.events import EventBus  # noqa: E402
from lofop.registries import SCHEDULERS  # noqa: E402
from lofop.training import build_scheduler  # noqa: E402
from lofop.training.schedulers import warmup_cosine  # noqa: E402
from lofop.training.trainer import warmup_cosine_lr  # noqa: E402


def _optimizer(lr=1.0):
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


class TestSchedulers:
    def test_all_registered(self):
        for name in ("warmup_cosine", "warmup_linear", "constant", "step"):
            assert name in SCHEDULERS

    def test_warmup_cosine_matches_legacy_multiplier(self):
        # The registered default must reproduce the historical schedule.
        opt = _optimizer()
        sched = warmup_cosine(opt, warmup_steps=2, total_steps=10)
        got = []
        for _ in range(10):
            got.append(opt.param_groups[0]["lr"])
            sched.step()
        expected = [
            warmup_cosine_lr(step, warmup_steps=2, total_steps=10) for step in range(10)
        ]
        assert got == pytest.approx(expected)

    def test_warmup_ramps_up(self):
        opt = _optimizer()
        sched = build_scheduler("warmup_linear", opt, warmup_steps=4, total_steps=20)
        first = opt.param_groups[0]["lr"]
        sched.step()
        assert opt.param_groups[0]["lr"] > first  # still warming up

    def test_constant_holds_peak_after_warmup(self):
        opt = _optimizer(lr=0.5)
        sched = build_scheduler("constant", opt, warmup_steps=1, total_steps=10)
        sched.step()  # past warmup
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)

    def test_step_decay_drops_at_milestone(self):
        opt = _optimizer(lr=1.0)
        sched = build_scheduler(
            "step", opt, warmup_steps=0, total_steps=10, gamma=0.1, milestones=[0.5],
        )
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


class TestEarlyStopping:
    def test_stops_when_no_improvement(self, tmp_path):
        from lofop.data.synthetic import generate_shapes_dataset
        from lofop.models import ApexHead, DeltaFusion, LofopDetect, RidgeNet
        from lofop.training import DetectionTorchDataset, Trainer

        root = tmp_path / "es"
        train = DetectionTorchDataset(
            generate_shapes_dataset(root / "t", num_images=8, image_size=64, seed=1),
            image_size=64,
        )
        val = DetectionTorchDataset(
            generate_shapes_dataset(root / "v", num_images=4, image_size=64, seed=2),
            image_size=64,
        )
        backbone = RidgeNet(widths=(16, 32, 64, 128), depths=(1, 1, 1, 1))
        model = LofopDetect(
            backbone,
            DeltaFusion(in_channels=backbone.out_channels, width=32),
            ApexHead(num_classes=2, width=32, num_convs=1),
            score_threshold=0.05,
        )
        bus = EventBus()
        stopped = {"epoch": None}
        bus.subscribe("train.early_stop", lambda e: stopped.update(epoch=e["epoch"]))
        epochs_seen = []
        bus.subscribe("train.epoch_end", lambda e: epochs_seen.append(e["epoch"]))
        # min_delta huge => no epoch ever "improves" => stop after `patience`.
        trainer = Trainer(
            model, train, val, epochs=10, batch_size=4, workers=0,
            checkpoint_dir=root / "run", device="cpu", events=bus,
            patience=2, min_delta=10.0,
        )
        trainer.fit()
        assert stopped["epoch"] is not None
        assert len(epochs_seen) < 10  # stopped before the full run


class TestTensorBoardHook:
    def test_logs_scalars(self, tmp_path):
        pytest.importorskip("torch.utils.tensorboard")
        from lofop.training import attach_tensorboard

        bus = EventBus()
        hook = attach_tensorboard(tmp_path / "tb", events=bus)
        bus.emit(
            "train.epoch_end", epoch=0, loss=1.5,
            metrics={"map50": 0.4, "map50_95": 0.2, "precision": 0.5,
                     "recall": 0.6, "f1": 0.55},
        )
        hook.close()
        assert list((tmp_path / "tb").glob("events.out.tfevents.*"))
