from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LicenseOptions:
    enabled: bool = False
    app_id: str = ""
    app_key: str = ""
    encrypt_key: str = ""
    host_url: str = ""
    request_timeout_sec: int = 5
    bind_hardware: bool = True
    remember_card_default: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LicenseOptions":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            app_id=str(data.get("app_id", "")),
            app_key=str(data.get("app_key", "")),
            encrypt_key=str(data.get("encrypt_key", "")),
            host_url=str(data.get("host_url", "")),
            request_timeout_sec=max(1, int(data.get("request_timeout_sec", 5))),
            bind_hardware=bool(data.get("bind_hardware", True)),
            remember_card_default=bool(data.get("remember_card_default", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""

        missing_fields = [
            field_name
            for field_name, value in (
                ("app_id", self.app_id),
                ("app_key", self.app_key),
                ("encrypt_key", self.encrypt_key),
                ("host_url", self.host_url),
            )
            if not value.strip()
        ]
        if missing_fields:
            joined = "、".join(missing_fields)
            return False, f"授权配置缺少必填项：{joined}"

        key_length = len(self.encrypt_key.encode("utf-8"))
        if key_length not in {16, 24, 32}:
            return False, "encrypt_key 长度必须为 16、24 或 32 字节"

        return True, ""


@dataclass
class LicenseState:
    remember_card: bool = False
    card_ciphertext: str = ""
    device_fingerprint: str = ""
    fallback_device_id: str = ""
    last_login_at: str = ""
    last_expire_at: str = ""
    last_card_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LicenseState":
        data = data or {}
        return cls(
            remember_card=bool(data.get("remember_card", False)),
            card_ciphertext=str(data.get("card_ciphertext", "")),
            device_fingerprint=str(data.get("device_fingerprint", "")),
            fallback_device_id=str(data.get("fallback_device_id", "")),
            last_login_at=str(data.get("last_login_at", "")),
            last_expire_at=str(data.get("last_expire_at", "")),
            last_card_info=dict(data.get("last_card_info", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
