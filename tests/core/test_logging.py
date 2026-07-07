"""Tests for lofop.core.logging and lofop.core.exceptions."""

import logging

import pytest

import lofop.core.logging as lofop_logging
from lofop.core.exceptions import ConfigError, EventError, LofopError
from lofop.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_logging_state():
    yield
    root = logging.getLogger("lofop")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    lofop_logging._configured = False


class TestGetLogger:
    def test_namespacing(self):
        assert get_logger().name == "lofop"
        assert get_logger("lofop").name == "lofop"
        assert get_logger("data.loader").name == "lofop.data.loader"
        assert get_logger("lofop.core.registry").name == "lofop.core.registry"


class TestConfigureLogging:
    def test_idempotent_without_force(self):
        logger = configure_logging(level="DEBUG", use_rich=False)
        count = len(logger.handlers)
        configure_logging(use_rich=False)
        assert len(logger.handlers) == count
        assert logger.level == logging.DEBUG

    def test_force_reconfigures(self):
        configure_logging(level="INFO", use_rich=False)
        logger = configure_logging(level="WARNING", use_rich=False, force=True)
        assert logger.level == logging.WARNING

    def test_env_level(self, monkeypatch):
        monkeypatch.setenv("LOFOP_LOG_LEVEL", "ERROR")
        assert configure_logging(use_rich=False).level == logging.ERROR

    def test_invalid_level_rejected(self):
        with pytest.raises(ValueError):
            configure_logging(level="LOUD", use_rich=False)

    def test_log_file_receives_records(self, tmp_path):
        log_file = tmp_path / "logs" / "run.log"
        configure_logging(level="INFO", log_file=log_file, use_rich=False)
        get_logger("test").debug("into the file")
        for handler in logging.getLogger("lofop").handlers:
            handler.flush()
        assert "into the file" in log_file.read_text()


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ConfigError, LofopError)
        with pytest.raises(LofopError):
            raise ConfigError("nope")

    def test_context_rendered(self):
        err = ConfigError("Missing key", context={"path": "model.depth"})
        assert "Missing key" in str(err)
        assert "model.depth" in str(err)
        assert err.context == {"path": "model.depth"}

    def test_event_error_carries_failures(self):
        failures = [("handler_a", RuntimeError("boom"))]
        err = EventError("failed", failures=failures)
        assert err.failures == failures
