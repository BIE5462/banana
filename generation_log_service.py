from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config import get_generation_log_path


@dataclass
class GenerationLogItem:
    group_index: int
    match_key: str
    prompt_text: str
    variant_index: int
    success: bool
    reference_images: list[str] = field(default_factory=list)
    saved_paths: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = 0.0
    request_seconds: float = 0.0
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationLogItem":
        if not isinstance(data, dict):
            raise ValueError("日志明细格式无效。")
        return cls(
            group_index=int(data.get("group_index", 0)),
            match_key=str(data.get("match_key", "")),
            prompt_text=str(data.get("prompt_text", "")),
            variant_index=int(data.get("variant_index", 0)),
            success=bool(data.get("success", False)),
            reference_images=_to_string_list(data.get("reference_images")),
            saved_paths=_to_string_list(data.get("saved_paths")),
            error=str(data.get("error", "")),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            request_seconds=float(data.get("request_seconds", 0.0)),
            seed=_to_optional_int(data.get("seed")),
        )

    def to_dict(self) -> dict:
        return {
            "group_index": self.group_index,
            "match_key": self.match_key,
            "prompt_text": self.prompt_text,
            "variant_index": self.variant_index,
            "success": self.success,
            "reference_images": list(self.reference_images),
            "saved_paths": list(self.saved_paths),
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "request_seconds": self.request_seconds,
            "seed": self.seed,
        }


@dataclass
class GenerationLogEntry:
    id: str
    started_at: str
    finished_at: str
    status: str
    total_planned: int
    completed: int
    success_count: int
    failure_count: int
    stopped: bool
    output_dir: str
    api_base_url: str
    model_type: str
    temperature: float
    top_p: float
    aspect_ratio: str
    image_size: str
    variants_per_group: int
    items: list[GenerationLogItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationLogEntry":
        if not isinstance(data, dict):
            raise ValueError("日志记录格式无效。")
        entry_id = str(data.get("id", "")).strip()
        started_at = str(data.get("started_at", "")).strip()
        finished_at = str(data.get("finished_at", "")).strip()
        if not entry_id or not started_at or not finished_at:
            raise ValueError("日志记录缺少必要字段。")
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("日志明细列表格式无效。")
        return cls(
            id=entry_id,
            started_at=started_at,
            finished_at=finished_at,
            status=str(data.get("status", "failed")),
            total_planned=int(data.get("total_planned", 0)),
            completed=int(data.get("completed", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            stopped=bool(data.get("stopped", False)),
            output_dir=str(data.get("output_dir", "")),
            api_base_url=str(data.get("api_base_url", "")),
            model_type=str(data.get("model_type", "")),
            temperature=float(data.get("temperature", 0.0)),
            top_p=float(data.get("top_p", 0.0)),
            aspect_ratio=str(data.get("aspect_ratio", "")),
            image_size=str(data.get("image_size", "")),
            variants_per_group=int(data.get("variants_per_group", 1)),
            items=[GenerationLogItem.from_dict(item) for item in raw_items],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "total_planned": self.total_planned,
            "completed": self.completed,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "stopped": self.stopped,
            "output_dir": self.output_dir,
            "api_base_url": self.api_base_url,
            "model_type": self.model_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "aspect_ratio": self.aspect_ratio,
            "image_size": self.image_size,
            "variants_per_group": self.variants_per_group,
            "items": [item.to_dict() for item in self.items],
        }


def load_entries(log_path: str | Path | None = None) -> list[GenerationLogEntry]:
    path = Path(log_path) if log_path else get_generation_log_path()
    return _load_entries_from_path(path)


def append_entry(
    entry: GenerationLogEntry, log_path: str | Path | None = None
) -> Path:
    path = Path(log_path) if log_path else get_generation_log_path()
    entries = _load_entries_from_path(path)
    entries.insert(0, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.to_dict() for item in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _load_entries_from_path(path: Path) -> list[GenerationLogEntry]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    entries: list[GenerationLogEntry] = []
    for item in data:
        try:
            entries.append(GenerationLogEntry.from_dict(item))
        except (TypeError, ValueError):
            continue
    return entries


def _to_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _to_optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
