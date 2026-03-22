from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests


class ApiUsageError(Exception):
    """额度查询失败时抛出的异常。"""


@dataclass
class ApiUsageSnapshot:
    name: str
    total_available: float
    total_granted: float
    total_used: float
    expires_at: int
    model_limits: dict[str, Any] = field(default_factory=dict)
    model_limits_enabled: bool = False
    unlimited_quota: bool = False
    fetched_at: datetime = field(default_factory=datetime.now)
    base_url: str = ""

    @property
    def remaining_quota(self) -> float:
        return self.total_available


def build_usage_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ApiUsageError("请先填写 URL。")
    return f"{normalized}/api/usage/token/"


def format_expires_at(expires_at: int) -> str:
    if expires_at <= 0:
        return "不过期或未提供"
    try:
        return datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "时间格式无效"


def format_fetched_at(value: datetime | None) -> str:
    if value is None:
        return "未刷新"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_model_limits(model_limits: dict[str, Any]) -> str:
    if not model_limits:
        return "未提供模型限制"

    lines: list[str] = []
    for model_name, limit in sorted(model_limits.items(), key=lambda item: str(item[0])):
        lines.append(f"{model_name}: {limit}")
    return "\n".join(lines)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fetch_token_usage(
    base_url: str,
    api_key: str,
    timeout: int = 10,
    session: requests.Session | None = None,
) -> ApiUsageSnapshot:
    normalized_key = api_key.strip()
    if not normalized_key:
        raise ApiUsageError("请先填写 API Key。")

    url = build_usage_url(base_url)
    client = session or requests

    try:
        response = client.get(
            url,
            headers={"Authorization": f"Bearer {normalized_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiUsageError(f"额度查询失败：{exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiUsageError("额度接口返回了无效的 JSON 数据。") from exc

    if not isinstance(payload, dict):
        raise ApiUsageError("额度接口返回格式无效。")

    success_flag = payload.get("code")
    if success_flag is None:
        success_flag = payload.get("success")

    if not success_flag:
        message = str(payload.get("message") or "额度查询失败。")
        raise ApiUsageError(message)

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ApiUsageError("额度数据缺失。")

    model_limits = data.get("model_limits")
    if not isinstance(model_limits, dict):
        model_limits = {}

    return ApiUsageSnapshot(
        name=str(data.get("name") or ""),
        total_available=_to_float(data.get("total_available")),
        total_granted=_to_float(data.get("total_granted")),
        total_used=_to_float(data.get("total_used")),
        expires_at=_to_int(data.get("expires_at")),
        model_limits=model_limits,
        model_limits_enabled=bool(data.get("model_limits_enabled", False)),
        unlimited_quota=bool(data.get("unlimited_quota", False)),
        fetched_at=datetime.now(),
        base_url=base_url.strip(),
    )
