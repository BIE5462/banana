from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from license_models import LicenseOptions, LicenseState

EMBEDDED_LICENSE_OPTIONS = LicenseOptions(
    enabled=True,
    app_id="2154",#"482",
    app_key="8068250322980071",
    encrypt_key="ii8ZwSkwtr8feNSN",
    host_url="https://www.wlyz.cn/api",
    request_timeout_sec=5,
    bind_hardware=True,
    remember_card_default=True,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_license_state_dir() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return ensure_dir(Path(appdata) / "NanoBananaBatch")
    return ensure_dir(Path.home() / ".nanobanana-batch")


def get_license_state_file() -> Path:
    return get_license_state_dir() / "license_state.json"


def read_json_file(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json_file(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def load_license_state(state_path: str | Path | None = None) -> LicenseState:
    path = Path(state_path) if state_path else get_license_state_file()
    data = read_json_file(path, {})
    if not isinstance(data, dict):
        return LicenseState()
    return LicenseState.from_dict(data)


def save_license_state(
    state: LicenseState, state_path: str | Path | None = None
) -> Path:
    path = Path(state_path) if state_path else get_license_state_file()
    write_json_file(path, state.to_dict())
    return path


def load_license_options(
    config_path: str | Path | None = None,
) -> tuple[bool, str, LicenseOptions]:
    del config_path

    valid, message = EMBEDDED_LICENSE_OPTIONS.validate()
    if not valid:
        return False, f"内置授权配置无效：{message}", LicenseOptions()

    return True, "已加载内置授权配置。", EMBEDDED_LICENSE_OPTIONS
