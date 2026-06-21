"""
Tests for GET /settings and PUT /settings endpoints.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import config
import server as srv


@pytest.fixture()
def client(tmp_path):
    outputs = tmp_path / "outputs"
    archive = outputs / "archive"
    pdfs    = tmp_path / "pdfs"
    outputs.mkdir(); archive.mkdir(); pdfs.mkdir()
    settings_path = tmp_path / "settings.json"
    with (
        patch.object(srv, "_OUTPUTS_DIR",   outputs),
        patch.object(srv, "_ARCHIVE_DIR",   archive),
        patch.object(srv, "_PDFS_DIR",      pdfs),
        patch.object(srv, "_SETTINGS_PATH", settings_path),
        patch("server.importlib.reload"),
    ):
        app = srv.create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, settings_path


class TestGetSettings:
    def test_returns_200(self, client):
        c, _ = client
        assert c.get("/settings").status_code == 200

    def test_response_has_provider_key(self, client):
        c, _ = client
        assert "PROVIDER" in c.get("/settings").json()

    def test_response_has_models_key(self, client):
        c, _ = client
        assert "MODELS" in c.get("/settings").json()

    def test_models_is_dict(self, client):
        c, _ = client
        assert isinstance(c.get("/settings").json()["MODELS"], dict)

    def test_provider_matches_config(self, client):
        c, _ = client
        assert c.get("/settings").json()["PROVIDER"] == config.PROVIDER

    def test_models_match_config(self, client):
        c, _ = client
        assert c.get("/settings").json()["MODELS"] == config.MODELS


class TestPutSettings:
    def _payload(self, provider="anthropic", models=None):
        if models is None:
            models = {"anthropic": "claude-sonnet-4-20250514"}
        return {"PROVIDER": provider, "MODELS": models}

    def test_returns_200_for_valid_payload(self, client):
        c, _ = client
        assert c.put("/settings", json=self._payload()).status_code == 200

    def test_returns_saved_status(self, client):
        c, _ = client
        assert c.put("/settings", json=self._payload()).json() == {"status": "saved"}

    def test_writes_settings_json_file(self, client):
        c, path = client
        c.put("/settings", json=self._payload())
        assert path.exists()

    def test_written_file_contains_correct_provider(self, client):
        c, path = client
        c.put("/settings", json=self._payload())
        assert json.loads(path.read_text())["PROVIDER"] == "anthropic"

    def test_written_file_contains_correct_models(self, client):
        c, path = client
        payload = self._payload()
        c.put("/settings", json=payload)
        assert json.loads(path.read_text())["MODELS"] == payload["MODELS"]

    def test_provider_not_in_models_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json={"PROVIDER": "missing", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}})
        assert r.status_code == 422

    def test_empty_models_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json={"PROVIDER": "anthropic", "MODELS": {}})
        assert r.status_code == 422

    def test_missing_provider_field_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json={"MODELS": {"anthropic": "claude-sonnet-4-20250514"}})
        assert r.status_code == 422

    def test_missing_models_field_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json={"PROVIDER": "anthropic"})
        assert r.status_code == 422

    def test_can_save_multiple_providers(self, client):
        c, path = client
        payload = {
            "PROVIDER": "google",
            "MODELS": {
                "google": "gemini-3-flash-preview",
                "anthropic": "claude-sonnet-4-20250514",
            },
        }
        r = c.put("/settings", json=payload)
        assert r.status_code == 200
        saved = json.loads(path.read_text())
        assert len(saved["MODELS"]) == 2

    def test_can_save_custom_model_key(self, client):
        c, path = client
        payload = {
            "PROVIDER": "google-pro",
            "MODELS": {"google-pro": "gemini-2.5-pro"},
        }
        r = c.put("/settings", json=payload)
        assert r.status_code == 200
        assert json.loads(path.read_text())["MODELS"]["google-pro"] == "gemini-2.5-pro"

    def test_settings_file_is_valid_json(self, client):
        c, path = client
        c.put("/settings", json=self._payload())
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


class TestDuplicatePolicySetting:
    def _payload(self, policy=None):
        payload = {"PROVIDER": "anthropic", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}}
        if policy is not None:
            payload["PIPELINE"] = {"duplicate_policy": policy}
        return payload

    def test_get_settings_includes_duplicate_policy(self, client):
        c, _ = client
        data = c.get("/settings").json()
        assert data["PIPELINE"]["duplicate_policy"] == config.PIPELINE["duplicate_policy"]

    def test_put_persists_duplicate_policy(self, client):
        c, path = client
        r = c.put("/settings", json=self._payload("keep_both"))
        assert r.status_code == 200
        saved = json.loads(path.read_text())
        assert saved["PIPELINE"] == {"duplicate_policy": "keep_both"}

    def test_put_without_pipeline_still_succeeds(self, client):
        c, path = client
        r = c.put("/settings", json=self._payload())
        assert r.status_code == 200
        assert "PIPELINE" not in json.loads(path.read_text())

    def test_put_invalid_policy_value_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json=self._payload("sometimes"))
        assert r.status_code == 422

    def test_put_unknown_pipeline_key_returns_422(self, client):
        c, _ = client
        payload = self._payload()
        payload["PIPELINE"] = {"max_review_cycles": "99"}
        r = c.put("/settings", json=payload)
        assert r.status_code == 422


class TestExtractorSetting:
    def _payload(self, extractor=None):
        payload = {"PROVIDER": "anthropic", "MODELS": {"anthropic": "claude-sonnet-4-20250514"}}
        if extractor is not None:
            payload["PIPELINE"] = {"extractor": extractor}
        return payload

    def test_get_settings_includes_extractor(self, client):
        c, _ = client
        data = c.get("/settings").json()
        assert data["PIPELINE"]["extractor"] == config.PIPELINE["extractor"]

    def test_put_persists_extractor(self, client):
        c, path = client
        r = c.put("/settings", json=self._payload("mineru"))
        assert r.status_code == 200
        saved = json.loads(path.read_text())
        assert saved["PIPELINE"] == {"extractor": "mineru"}

    def test_put_invalid_extractor_value_returns_422(self, client):
        c, _ = client
        r = c.put("/settings", json=self._payload("nope"))
        assert r.status_code == 422

    def test_invalid_policy_leaves_settings_file_unwritten(self, client):
        c, path = client
        c.put("/settings", json=self._payload("sometimes"))
        assert not path.exists()
