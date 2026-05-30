import importlib
from typing import Any


def optional_import(module_name: str, attr: str | None = None) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr) if attr else module
