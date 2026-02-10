import json
from pathlib import Path

from agent.config import load_config
from web.app import create_app


def _make_app(tmp_path: Path, config_data: dict):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    config = load_config(str(config_path))
    app = create_app(config)
    app.testing = True
    app.config["config_path"] = str(config_path)
    return app


def test_settings_raw_roundtrip(tmp_path):
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "chat_model": {"model_name": "llama3.2"},
        "extensions": {"memory_recall": False},
    }
    app = _make_app(tmp_path, config_data)
    client = app.test_client()

    resp = client.get("/api/settings/raw")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["config"]["chat_model"]["model_name"] == "llama3.2"


def test_settings_merge(tmp_path):
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "chat_model": {"model_name": "llama3.2"},
        "max_monologue_iterations": 25,
    }
    app = _make_app(tmp_path, config_data)
    client = app.test_client()

    resp = client.post("/api/settings", json={"max_monologue_iterations": 10})
    assert resp.status_code == 200

    updated = json.loads((tmp_path / "config.json").read_text())
    assert updated["chat_model"]["model_name"] == "llama3.2"
    assert updated["max_monologue_iterations"] == 10


def test_extensions_list_and_update(tmp_path):
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "extensions": {"memory_recall": False},
    }
    app = _make_app(tmp_path, config_data)
    client = app.test_client()

    resp = client.get("/api/extensions")
    assert resp.status_code == 200
    data = resp.get_json()
    names = {e["name"] for e in data["extensions"]}
    assert "memory_recall" in names

    resp = client.post("/api/extensions", json={"extensions": {"output_logger": False}})
    assert resp.status_code == 200
    updated = json.loads((tmp_path / "config.json").read_text())
    assert updated["extensions"]["output_logger"] is False


def test_models_pull_validation(tmp_path):
    app = _make_app(tmp_path, {"data_dir": str(tmp_path / "data")})
    client = app.test_client()
    resp = client.post("/api/models/pull", json={})
    assert resp.status_code == 400


def test_memory_delete_requires_id(tmp_path):
    app = _make_app(tmp_path, {"data_dir": str(tmp_path / "data")})
    client = app.test_client()
    resp = client.post("/api/memory/delete", json={})
    assert resp.status_code == 400


def test_rename_requires_title(tmp_path):
    app = _make_app(tmp_path, {"data_dir": str(tmp_path / "data")})
    client = app.test_client()
    resp = client.patch("/api/chat/session/abc123", json={})
    assert resp.status_code == 400
