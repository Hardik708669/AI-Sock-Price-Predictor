import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.core.config import get_settings


class ModelRegistry:
    def __init__(self, base_path: str | None = None) -> None:
        self.base_path = Path(base_path or get_settings().model_registry_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "registry.json"

    def register(self, name: str, model: Any, metrics: dict, metadata: dict | None = None) -> dict:
        version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        model_path = self.base_path / f"{name.replace(' ', '_').lower()}_{version}.joblib"
        joblib.dump(model, model_path)
        record = {"name": name, "version": version, "path": str(model_path), "metrics": metrics, "metadata": metadata or {}, "created_at": datetime.now(timezone.utc).isoformat()}
        registry = self.list_models()
        registry.append(record)
        self.index_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return record

    def list_models(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def latest(self, name: str) -> dict | None:
        records = [record for record in self.list_models() if record["name"] == name]
        return sorted(records, key=lambda item: item["version"])[-1] if records else None
