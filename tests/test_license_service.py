from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from license_crypto import encrypt_aes_ecb_pkcs5
from license_models import LicenseOptions
from license_service import LicenseManager
from license_store import load_license_state


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class LicenseServiceTests(unittest.TestCase):
    def build_options(self, *, bind_hardware: bool = False) -> LicenseOptions:
        return LicenseOptions(
            enabled=True,
            app_id="demo-app",
            app_key="demo-app-key",
            encrypt_key="1234567890abcdef",
            host_url="https://license.example.com",
            request_timeout_sec=5,
            bind_hardware=bind_hardware,
            remember_card_default=True,
        )

    def test_build_request_url_contains_expected_plain_params(self) -> None:
        manager = LicenseManager(self.build_options(bind_hardware=False))
        with patch("license_service.time.time", return_value=1700000000.123), patch(
            "license_service.random.randint", return_value=1234567890123456
        ):
            request_url = manager.build_request_url(
                "login",
                {"card": "CARD-001", "mac": "DEVICE-001"},
            )

        plain_params = manager.decode_request_params(request_url)
        self.assertIn("card=CARD-001", plain_params)
        self.assertIn("mac=DEVICE-001", plain_params)
        self.assertIn("safeCode=1234567890123456", plain_params)
        self.assertIn("timestamp=1700000000123", plain_params)
        self.assertIn("signature=", plain_params)

    def test_login_success_updates_state_and_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "license_state.json"
            with patch.object(
                LicenseManager, "_run_powershell", return_value="DEVICE-ID"
            ):
                manager = LicenseManager(
                    self.build_options(bind_hardware=True), state_path=state_path
                )

            with patch.object(
                manager,
                "_request",
                return_value=(
                    True,
                    "ok",
                    {"token": "token-001", "expireTime": "2026-12-31 23:59:59"},
                ),
            ):
                success, _, data = manager.login(" CARD-001 ", remember_card=True)

            saved_state = load_license_state(state_path)
            self.assertTrue(success)
            self.assertEqual(manager.token, "token-001")
            self.assertEqual(manager.current_card, "CARD-001")
            self.assertTrue(saved_state.remember_card)
            self.assertNotEqual(saved_state.card_ciphertext, "")
            self.assertEqual(data["token"], "token-001")

    def test_login_failure_keeps_session_empty(self) -> None:
        manager = LicenseManager(self.build_options(bind_hardware=False))
        with patch.object(
            manager,
            "_request",
            return_value=(False, "卡密已过期", None),
        ):
            success, message, data = manager.login("CARD-001")

        self.assertFalse(success)
        self.assertEqual(message, "卡密已过期")
        self.assertIsNone(data)
        self.assertFalse(manager.is_logged_in)

    def test_device_fingerprint_uses_fallback_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "license_state.json"
            with patch.object(LicenseManager, "_run_powershell", return_value=""):
                manager = LicenseManager(
                    self.build_options(bind_hardware=True), state_path=state_path
                )

        self.assertNotEqual(manager.state.fallback_device_id, "")
        self.assertEqual(len(manager.state.device_fingerprint), 64)

    def test_request_raw_maps_timeout(self) -> None:
        manager = LicenseManager(self.build_options(bind_hardware=False))

        class TimeoutClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self) -> "TimeoutClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def get(self, url: str) -> _FakeResponse:
                raise httpx.TimeoutException("timeout")

        with patch("license_service.httpx.Client", TimeoutClient):
            success, message, data = manager._request_raw("login", {"card": "CARD"})

        self.assertFalse(success)
        self.assertEqual(message, "网络请求超时")
        self.assertIsNone(data)

    def test_request_raw_rejects_non_dict_response(self) -> None:
        manager = LicenseManager(self.build_options(bind_hardware=False))
        encrypted_text = encrypt_aes_ecb_pkcs5(
            json.dumps(["bad", "payload"]), manager.options.encrypt_key
        )

        class SuccessClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self) -> "SuccessClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def get(self, url: str) -> _FakeResponse:
                return _FakeResponse(200, encrypted_text)

        with patch("license_service.httpx.Client", SuccessClient):
            success, message, data = manager._request_raw("login", {"card": "CARD"})

        self.assertFalse(success)
        self.assertEqual(message, "授权响应格式异常")
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
