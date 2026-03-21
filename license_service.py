from __future__ import annotations

import json
import random
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from license_crypto import (
    decrypt_aes_ecb_pkcs5,
    encrypt_aes_ecb_pkcs5,
    encrypt_text,
    sha256_hex,
    try_decrypt_text,
)
from license_models import LicenseOptions, LicenseState
from license_store import load_license_state, save_license_state

ERROR_CODES = {
    "-10002": "卡密已过期",
    "-10013": "卡密已被禁用",
    "-10014": "卡密不存在",
    "-10034": "应用不可用",
    "-10035": "用户不在线",
    "-10038": "应用已停用",
    "-10045": "授权已失效",
}


class LicenseManager:
    def __init__(
        self,
        options: LicenseOptions,
        state_path: str | Path | None = None,
    ) -> None:
        self.options = options
        self.state_path = Path(state_path) if state_path else None
        self.state = self._load_state()
        self.token = ""
        self.card_info: dict[str, Any] = {}
        self.current_card = ""
        self.current_mac = ""

        if self.options.bind_hardware and not self.state.device_fingerprint:
            self.state.device_fingerprint = self._build_device_fingerprint()
            self._save_state()

    @property
    def is_enabled(self) -> bool:
        return self.options.enabled

    @property
    def is_logged_in(self) -> bool:
        return bool(self.token)

    def get_remembered_card(self) -> str:
        if not self.state.remember_card or not self.state.card_ciphertext:
            return ""
        success, plain_text = try_decrypt_text(self.state.card_ciphertext)
        return plain_text if success else ""

    def login(
        self, card: str, remember_card: bool | None = None
    ) -> tuple[bool, str, dict[str, Any] | None]:
        if not self.options.enabled:
            return True, "未启用卡密验证", {}

        remember_card = (
            self.options.remember_card_default
            if remember_card is None
            else remember_card
        )
        normalized_card = card.strip()
        if not normalized_card:
            return False, "请输入卡密", None

        mac = ""
        if self.options.bind_hardware:
            mac = self.state.device_fingerprint or self._build_device_fingerprint()

        success, message, data = self._request(
            "login",
            {"card": normalized_card, "mac": mac},
        )
        if not success:
            return False, message, None

        self.current_card = normalized_card
        self.current_mac = mac
        self.token = str((data or {}).get("token", ""))
        self.card_info = dict(data or {})
        self.state.remember_card = bool(remember_card)
        self.state.card_ciphertext = (
            encrypt_text(normalized_card) if remember_card else ""
        )
        self.state.last_login_at = datetime.now().isoformat(timespec="seconds")
        self.state.last_expire_at = self._extract_expire_text(self.card_info)
        self.state.last_card_info = self.card_info
        self._save_state()
        return True, "登录成功", self.card_info

    def logout(self, keep_saved_card: bool = True) -> tuple[bool, str]:
        if not self.token:
            self._clear_session(keep_saved_card=keep_saved_card)
            return True, "已退出"

        success, message, _ = self._request("logout", {"token": self.token})
        self._clear_session(keep_saved_card=keep_saved_card)
        return success, message

    def build_request_url(
        self,
        action: str,
        business_params: dict[str, Any],
        *,
        signature_in_params: bool = False,
    ) -> str:
        params = {key: value for key, value in business_params.items() if value != ""}
        self._extend_params(params)
        query = self._build_query_string(params)
        signature = sha256_hex(query + self.options.app_key)

        if signature_in_params:
            params["signature"] = signature
            plain_params = self._build_query_string(params)
        else:
            plain_params = f"{query}&signature={signature}"

        encrypted_params = encrypt_aes_ecb_pkcs5(
            plain_params, self.options.encrypt_key
        )
        return (
            f"{self.options.host_url.rstrip('/')}/single/{action}"
            f"?appId={self.options.app_id}&params={encrypted_params}"
        )

    def decode_request_params(self, request_url: str) -> str:
        query = parse_qs(urlsplit(request_url).query)
        encrypted_params = query.get("params", [""])[0]
        return decrypt_aes_ecb_pkcs5(encrypted_params, self.options.encrypt_key)

    def _request(
        self, action: str, business_params: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        success, message, response_dict = self._request_raw(action, business_params)
        if not success:
            return False, message, None
        if response_dict.get("code") != 1:
            code = str(response_dict.get("code", ""))
            mapped_message = ERROR_CODES.get(code, str(response_dict.get("msg", "请求失败")))
            return False, mapped_message, None
        data = dict(response_dict.get("data", {}) or {})
        return True, str(response_dict.get("msg", "请求成功")), data

    def _request_raw(
        self,
        action: str,
        business_params: dict[str, Any],
        *,
        signature_in_params: bool = False,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        try:
            request_url = self.build_request_url(
                action,
                business_params,
                signature_in_params=signature_in_params,
            )
            with httpx.Client(timeout=self.options.request_timeout_sec) as client:
                response = client.get(request_url)

            if response.status_code != 200:
                return False, "网络连接异常", None

            decrypted_response = decrypt_aes_ecb_pkcs5(
                response.text, self.options.encrypt_key
            )
            response_dict = json.loads(decrypted_response)
            if not isinstance(response_dict, dict):
                return False, "授权响应格式异常", None
            return True, "", response_dict
        except httpx.TimeoutException:
            return False, "网络请求超时", None
        except httpx.HTTPError:
            return False, "网络连接异常", None
        except ValueError as exc:
            return False, str(exc), None
        except Exception:
            return False, "授权请求异常", None

    def _extend_params(self, params: dict[str, Any]) -> None:
        params["timestamp"] = int(time.time() * 1000)
        params["safeCode"] = random.randint(10**15, 10**16 - 1)

    def _build_query_string(self, params: dict[str, Any]) -> str:
        return urlencode(dict(sorted(params.items())))

    def _build_device_fingerprint(self) -> str:
        if self.state.device_fingerprint:
            return self.state.device_fingerprint

        components: list[str] = []
        commands = [
            ("CPU", "(Get-CimInstance Win32_Processor).ProcessorId"),
            ("Board", "(Get-CimInstance Win32_BaseBoard).SerialNumber"),
        ]
        for prefix, command in commands:
            value = self._run_powershell(command)
            if value and value not in {"None", "0", "To be filled by O.E.M."}:
                components.append(f"{prefix}:{value}")

        if not components:
            system_uuid = self._run_powershell(
                "(Get-CimInstance Win32_ComputerSystemProduct).UUID"
            )
            if system_uuid and system_uuid != "UUID":
                components.append(f"UUID:{system_uuid}")

        if not components:
            if not self.state.fallback_device_id:
                self.state.fallback_device_id = str(uuid.uuid4())
            components.append(f"FALLBACK:{self.state.fallback_device_id}")

        fingerprint = sha256_hex("_".join(components))
        self.state.device_fingerprint = fingerprint
        self._save_state()
        return fingerprint

    def _run_powershell(self, command: str) -> str:
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _extract_expire_text(self, info: dict[str, Any]) -> str:
        for key in ("endTime", "expireTime", "deadline", "dueTime"):
            value = info.get(key)
            if value:
                return str(value)
        return ""

    def _clear_session(self, keep_saved_card: bool) -> None:
        self.token = ""
        self.card_info = {}
        self.current_card = ""
        self.current_mac = ""
        if not keep_saved_card:
            self.state.remember_card = False
            self.state.card_ciphertext = ""
        self._save_state()

    def _load_state(self) -> LicenseState:
        return load_license_state(self.state_path)

    def _save_state(self) -> None:
        save_license_state(self.state, self.state_path)
