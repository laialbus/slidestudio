"""
Tests for config.py settings.json override loading.
"""
import importlib
import json
from pathlib import Path

import pytest

import config

_SETTINGS_PATH = Path(__file__).parent.parent / "settings.json"
_DEFAULT_PROVIDER = "google"


@pytest.fixture(autouse=True)
def clean_settings():
    existed  = _SETTINGS_PATH.exists()
    original = _SETTINGS_PATH.read_bytes() if existed else None
    _SETTINGS_PATH.unlink(missing_ok=True)
    yield
    _SETTINGS_PATH.unlink(missing_ok=True)
    if existed and original:
        _SETTINGS_PATH.write_bytes(original)
    importlib.reload(config)


class TestNoSettingsFile:
    def test_default_provider_is_used(self):
        importlib.reload(config)
        assert config.PROVIDER == _DEFAULT_PROVIDER

    def test_default_models_are_present(self):
        importlib.reload(config)
        assert "anthropic" in config.MODELS
        assert "google" in config.MODELS


class TestValidSettingsFile:
    def test_overrides_provider(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "anthropic", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}}),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert config.PROVIDER == "anthropic"

    def test_replaces_models(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "anthropic", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}}),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert config.MODELS == {"anthropic": "claude-sonnet-4-20250514"}

    def test_default_models_absent_after_replace(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "anthropic", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}}),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert "google" not in config.MODELS

    def test_custom_key_accepted(self):
        _SETTINGS_PATH.write_text(
            json.dumps({
                "PROVIDER": "google-pro",
                "MODELS": {"google-pro": "gemini-2.5-pro"},
            }),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert config.PROVIDER == "google-pro"
        assert config.MODELS["google-pro"] == "gemini-2.5-pro"

    def test_only_models_no_provider_keeps_default_provider(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"MODELS": {"google": "gemini-2.5-flash-lite"}}),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert config.PROVIDER == _DEFAULT_PROVIDER
        assert config.MODELS == {"google": "gemini-2.5-flash-lite"}

    def test_only_provider_no_models_keeps_default_models(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "google"}),
            encoding="utf-8",
        )
        importlib.reload(config)
        assert config.PROVIDER == "google"
        assert "anthropic" in config.MODELS


class TestMalformedSettingsFile:
    def test_malformed_json_raises_system_exit(self):
        _SETTINGS_PATH.write_text("not valid json {{", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(config)
        assert "malformed" in str(exc_info.value).lower()

    def test_malformed_error_message_mentions_settings_json(self):
        _SETTINGS_PATH.write_text("{invalid}", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(config)
        assert "settings.json" in str(exc_info.value)


class TestInvalidProviderInSettingsFile:
    def test_provider_not_in_models_raises_system_exit(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "nonexistent", "MODELS": {"google": "gemini-3-flash-preview"}}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(config)
        assert "nonexistent" in str(exc_info.value)

    def test_error_message_lists_available_keys(self):
        _SETTINGS_PATH.write_text(
            json.dumps({"PROVIDER": "bad-key", "MODELS": {"google": "gemini-3-flash-preview"}}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(config)
        assert "google" in str(exc_info.value)
