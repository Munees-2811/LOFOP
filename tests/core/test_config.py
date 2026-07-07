"""Tests for lofop.core.config."""

import pytest

from lofop.core.config import Config
from lofop.core.exceptions import ConfigError


class TestBasics:
    def test_attribute_and_item_access(self):
        cfg = Config({"model": {"depth": 50}, "lr": 0.1})
        assert cfg.lr == 0.1
        assert cfg["model"].depth == 50
        assert isinstance(cfg.model, Config)

    def test_missing_key(self):
        cfg = Config()
        with pytest.raises(AttributeError):
            _ = cfg.nope
        with pytest.raises(KeyError):
            _ = cfg["nope"]

    def test_input_not_aliased(self):
        source = {"model": {"depth": 50}}
        cfg = Config(source)
        source["model"]["depth"] = 99
        assert cfg.model.depth == 50

    def test_select_and_update_path(self):
        cfg = Config({"model": {"backbone": {"depth": 50}}})
        assert cfg.select("model.backbone.depth") == 50
        assert cfg.select("model.nope", default=None) is None
        with pytest.raises(ConfigError):
            cfg.select("model.nope")
        cfg.update_path("data.loader.workers", 8)
        assert cfg.data.loader.workers == 8

    def test_merge_deep(self):
        cfg = Config({"model": {"depth": 50, "width": 1.0}, "lr": 0.1})
        cfg.merge({"model": {"depth": 101}, "epochs": 12})
        assert cfg.model.depth == 101
        assert cfg.model.width == 1.0
        assert cfg.epochs == 12

    def test_to_dict_roundtrip(self):
        data = {"model": {"depth": 50}, "tags": ["a", "b"]}
        assert Config(data).to_dict() == data

    def test_freeze(self):
        cfg = Config({"model": {"depth": 50}})
        cfg.freeze()
        assert cfg.is_frozen
        with pytest.raises(ConfigError):
            cfg.lr = 0.1
        with pytest.raises(ConfigError):
            cfg.model.depth = 101
        with pytest.raises(ConfigError):
            del cfg["model"]


class TestYamlLoading:
    def test_load_and_dump(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("model:\n  depth: 50\nlr: 0.1\n")
        cfg = Config.load(path)
        assert cfg.model.depth == 50
        out = tmp_path / "out" / "cfg.yaml"
        cfg.dump(out)
        assert Config.load(out).to_dict() == cfg.to_dict()

    def test_extends_merges_parents_in_order(self, tmp_path):
        (tmp_path / "base.yaml").write_text("lr: 0.1\nmodel:\n  depth: 50\n  width: 1.0\n")
        (tmp_path / "sched.yaml").write_text("epochs: 12\nlr: 0.2\n")
        (tmp_path / "exp.yaml").write_text(
            "extends: [base.yaml, sched.yaml]\nmodel:\n  depth: 101\n"
        )
        cfg = Config.load(tmp_path / "exp.yaml")
        assert cfg.lr == 0.2  # later parent wins
        assert cfg.model.depth == 101  # child wins over parents
        assert cfg.model.width == 1.0
        assert cfg.epochs == 12
        assert "extends" not in cfg

    def test_extends_cycle_detected(self, tmp_path):
        (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
        with pytest.raises(ConfigError, match="Cyclic"):
            Config.load(tmp_path / "a.yaml")

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="Cannot read"):
            Config.load(tmp_path / "nope.yaml")

    def test_non_mapping_root_rejected(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ConfigError, match="mapping"):
            Config.load(path)

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("a: [1, 2\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            Config.load(path)


class TestInterpolation:
    def test_config_reference_preserves_type(self):
        cfg = Config({"base_lr": 0.1, "optimizer": {"lr": "${base_lr}"}})
        cfg.resolve()
        assert cfg.optimizer.lr == 0.1

    def test_embedded_reference_substitutes_text(self):
        cfg = Config({"name": "exp1", "workdir": "runs/${name}/ckpts"})
        cfg.resolve()
        assert cfg.workdir == "runs/exp1/ckpts"

    def test_env_reference(self, monkeypatch):
        monkeypatch.setenv("LOFOP_TEST_DATA", "/data")
        cfg = Config({"root": "${env:LOFOP_TEST_DATA}", "fallback": "${env:LOFOP_NOPE:none}"})
        cfg.resolve()
        assert cfg.root == "/data"
        assert cfg.fallback == "none"

    def test_env_missing_without_default(self, monkeypatch):
        monkeypatch.delenv("LOFOP_NOPE", raising=False)
        with pytest.raises(ConfigError, match="not set"):
            Config({"x": "${env:LOFOP_NOPE}"}).resolve()

    def test_unknown_reference(self):
        with pytest.raises(ConfigError, match="Unknown config reference"):
            Config({"x": "${nope.nope}"}).resolve()

    def test_cyclic_reference(self):
        with pytest.raises(ConfigError, match="Cyclic"):
            Config({"a": "${b}", "b": "${a}"}).resolve()

    def test_chained_references(self):
        cfg = Config({"a": "${b}", "b": "${c}", "c": 3})
        cfg.resolve()
        assert cfg.a == 3

    def test_references_inside_lists(self):
        cfg = Config({"size": 640, "scales": ["${size}", 320]})
        cfg.resolve()
        assert cfg.scales == [640, 320]

    def test_load_resolves_by_default(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("name: exp\nworkdir: runs/${name}\n")
        assert Config.load(path).workdir == "runs/exp"
        assert Config.load(path, resolve=False).workdir == "runs/${name}"
