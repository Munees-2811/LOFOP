"""Smoke test: the full training loop learns on synthetic shapes."""

import pytest

torch = pytest.importorskip("torch")

from lofop.core.events import EventBus  # noqa: E402
from lofop.data.synthetic import generate_shapes_dataset  # noqa: E402
from lofop.models import ApexHead, DeltaFusion, LofopDetect, RidgeNet  # noqa: E402
from lofop.training import DetectionTorchDataset, Trainer  # noqa: E402


def tiny_model():
    backbone = RidgeNet(widths=(16, 32, 64, 128), depths=(1, 1, 1, 1))
    neck = DeltaFusion(in_channels=backbone.out_channels, width=32)
    head = ApexHead(num_classes=2, width=32, num_convs=1)
    return LofopDetect(backbone, neck, head, score_threshold=0.05)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("shapes_train")
    train = generate_shapes_dataset(root / "train", num_images=16, image_size=64, seed=3)
    val = generate_shapes_dataset(root / "val", num_images=8, image_size=64, seed=4)
    return (
        DetectionTorchDataset(train, image_size=64, augment=True),
        DetectionTorchDataset(val, image_size=64),
    )


class TestTrainer:
    def test_fit_learns_checkpoints_and_events(self, data, tmp_path):
        train_ds, val_ds = data
        bus = EventBus()
        seen = {"epochs": [], "checkpoints": 0}
        bus.subscribe("train.epoch_end", lambda e: seen["epochs"].append(e["loss"]))
        bus.subscribe(
            "checkpoint.saved", lambda e: seen.update(checkpoints=seen["checkpoints"] + 1)
        )
        torch.manual_seed(0)
        trainer = Trainer(
            tiny_model(), train_ds, val_ds,
            epochs=3, batch_size=4, lr=0.02, warmup_epochs=0.5,
            workers=0, checkpoint_dir=tmp_path / "run", events=bus, device="cpu",
        )
        metrics = trainer.fit()

        assert len(seen["epochs"]) == 3
        assert seen["epochs"][-1] < seen["epochs"][0]  # loss went down
        assert seen["checkpoints"] == 3
        assert (tmp_path / "run" / "last.pt").is_file()
        assert (tmp_path / "run" / "best.pt").is_file()
        assert metrics is not None and 0.0 <= metrics.map50 <= 1.0

    def test_resume_restores_epoch_and_weights(self, data, tmp_path):
        train_ds, _ = data
        torch.manual_seed(0)
        first = Trainer(
            tiny_model(), train_ds, epochs=2, batch_size=4, workers=0,
            checkpoint_dir=tmp_path / "resume", device="cpu", events=EventBus(),
        )
        first.fit()

        second = Trainer(
            tiny_model(), train_ds, epochs=2, batch_size=4, workers=0,
            checkpoint_dir=tmp_path / "resume", device="cpu", events=EventBus(),
        )
        assert second.resume() == 2
        trained = first._unwrapped.state_dict()
        restored = second._unwrapped.state_dict()
        key = next(iter(trained))
        assert torch.equal(trained[key], restored[key])
