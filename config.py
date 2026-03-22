from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from GeminiImage import DEFAULT_MODEL_TYPE, DEFAULT_TIMEOUT


def get_default_config_path() -> Path:
    try:
        from PySide6.QtCore import QStandardPaths

        config_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        if config_dir:
            return Path(config_dir) / "config.json"
    except Exception:
        pass

    return Path.home() / ".nanobanana-batch" / "config.json"


def get_generation_log_path() -> Path:
    return get_default_config_path().with_name("generation_logs.json")


@dataclass
class FolderSlot:
    name: str = "参考图"
    path: str = ""
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "FolderSlot":
        return cls(
            name=str(data.get("name", "参考图")),
            path=str(data.get("path", "")),
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "enabled": self.enabled,
        }


@dataclass
class GenerationSettings:
    prompt: str = ""
    prompt_mode: str = "fixed"
    prompt_file_path: str = ""
    model_type: str = DEFAULT_MODEL_TYPE
    temperature: float = 0.8
    top_p: float = 0.65
    aspect_ratio: str = "Auto"
    image_size: str = "2K"
    timeout: int = DEFAULT_TIMEOUT
    variants_per_group: int = 1
    seed_enabled: bool = False
    base_seed: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationSettings":
        prompt_mode = str(data.get("prompt_mode", "fixed"))
        if prompt_mode not in {"fixed", "file"}:
            prompt_mode = "fixed"
        return cls(
            prompt=str(data.get("prompt", "")),
            prompt_mode=prompt_mode,
            prompt_file_path=str(data.get("prompt_file_path", "")),
            model_type=str(data.get("model_type", DEFAULT_MODEL_TYPE)),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.65)),
            aspect_ratio=str(data.get("aspect_ratio", "Auto")),
            image_size=str(data.get("image_size", "2K")),
            timeout=int(data.get("timeout", DEFAULT_TIMEOUT)),
            variants_per_group=max(1, int(data.get("variants_per_group", 1))),
            seed_enabled=bool(data.get("seed_enabled", False)),
            base_seed=int(data.get("base_seed", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "prompt_mode": self.prompt_mode,
            "prompt_file_path": self.prompt_file_path,
            "model_type": self.model_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "aspect_ratio": self.aspect_ratio,
            "image_size": self.image_size,
            "timeout": self.timeout,
            "variants_per_group": self.variants_per_group,
            "seed_enabled": self.seed_enabled,
            "base_seed": self.base_seed,
        }


@dataclass
class AppConfig:
    api_base_url: str = ""
    api_key: str = ""
    output_dir: str = ""
    folder_slots: list[FolderSlot] = field(default_factory=list)
    generation_settings: GenerationSettings = field(
        default_factory=GenerationSettings
    )

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(
            folder_slots=[
                FolderSlot(name="饰品"),
                FolderSlot(name="模特"),
                FolderSlot(name="手链"),
            ]
        )

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        folder_slots = [
            FolderSlot.from_dict(item)
            for item in data.get("folder_slots", [])
            if isinstance(item, dict)
        ]
        if not folder_slots:
            folder_slots = cls.default().folder_slots

        return cls(
            api_base_url=str(data.get("api_base_url", "")),
            api_key=str(data.get("api_key", "")),
            output_dir=str(data.get("output_dir", "")),
            folder_slots=folder_slots,
            generation_settings=GenerationSettings.from_dict(
                data.get("generation_settings", {})
            ),
        )

    def to_dict(self) -> dict:
        return {
            "api_base_url": self.api_base_url,
            "api_key": self.api_key,
            "output_dir": self.output_dir,
            "folder_slots": [slot.to_dict() for slot in self.folder_slots],
            "generation_settings": self.generation_settings.to_dict(),
        }


class ConfigManager:
    @staticmethod
    def load(config_path: str | Path | None = None) -> AppConfig:
        path = Path(config_path) if config_path else get_default_config_path()
        if not path.exists():
            return AppConfig.default()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppConfig.default()

        if not isinstance(data, dict):
            return AppConfig.default()
        return AppConfig.from_dict(data)

    @staticmethod
    def save(config: AppConfig, config_path: str | Path | None = None) -> Path:
        path = Path(config_path) if config_path else get_default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
