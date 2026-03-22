from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_usage_service import (
    ApiUsageError,
    build_usage_url,
    fetch_token_usage,
    format_expires_at,
    format_fetched_at,
    format_model_limits,
)


class _FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class _FakeSession:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    def get(self, url: str, headers: dict[str, str], timeout: int):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ApiUsageServiceTests(unittest.TestCase):
    def test_build_usage_url_trims_trailing_slash(self) -> None:
        self.assertEqual(
            build_usage_url("https://wuaiapi.com/"),
            "https://wuaiapi.com/api/usage/token/",
        )

    def test_fetch_token_usage_parses_snapshot(self) -> None:
        session = _FakeSession(
            _FakeResponse(
                {
                    "code": True,
                    "message": "ok",
                    "data": {
                        "name": "11",
                        "expires_at": 0,
                        "model_limits": {"gpt-4o": 10},
                        "model_limits_enabled": True,
                        "total_available": 100,
                        "total_granted": 120,
                        "total_used": 20,
                        "unlimited_quota": False,
                    },
                }
            )
        )

        snapshot = fetch_token_usage(
            "https://wuaiapi.com/",
            "sk-test",
            timeout=12,
            session=session,
        )

        self.assertEqual(
            session.calls[0]["url"],
            "https://wuaiapi.com/api/usage/token/",
        )
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"],
            "Bearer sk-test",
        )
        self.assertEqual(session.calls[0]["timeout"], 12)
        self.assertEqual(snapshot.name, "11")
        self.assertEqual(snapshot.total_available, 100)
        self.assertEqual(snapshot.total_granted, 120)
        self.assertEqual(snapshot.total_used, 20)
        self.assertEqual(snapshot.remaining_quota, 100)
        self.assertTrue(snapshot.model_limits_enabled)
        self.assertEqual(snapshot.model_limits["gpt-4o"], 10)

    def test_fetch_token_usage_rejects_failed_code(self) -> None:
        session = _FakeSession(
            _FakeResponse({"code": False, "message": "token 无效", "data": {}})
        )

        with self.assertRaisesRegex(ApiUsageError, "token 无效"):
            fetch_token_usage("https://wuaiapi.com", "sk-test", session=session)

    def test_fetch_token_usage_handles_invalid_json(self) -> None:
        session = _FakeSession(_FakeResponse(json_error=ValueError("bad json")))

        with self.assertRaisesRegex(ApiUsageError, "无效的 JSON"):
            fetch_token_usage("https://wuaiapi.com", "sk-test", session=session)

    def test_fetch_token_usage_handles_request_exception(self) -> None:
        session = _FakeSession(requests.ConnectionError("network down"))

        with self.assertRaisesRegex(ApiUsageError, "network down"):
            fetch_token_usage("https://wuaiapi.com", "sk-test", session=session)

    def test_fetch_token_usage_supports_success_flag_response(self) -> None:
        session = _FakeSession(
            _FakeResponse(
                {
                    "success": True,
                    "message": "",
                    "data": {
                        "name": "nano",
                        "expires_at": 0,
                        "model_limits": {},
                        "model_limits_enabled": False,
                        "total_available": 4996115551,
                        "total_granted": 5000000000,
                        "total_used": 3884449,
                        "unlimited_quota": False,
                    },
                }
            )
        )

        snapshot = fetch_token_usage(
            "https://api.qianhai.online",
            "sk-test",
            session=session,
        )

        self.assertEqual(snapshot.name, "nano")
        self.assertEqual(snapshot.total_available, 4996115551)
        self.assertEqual(snapshot.total_granted, 5000000000)
        self.assertEqual(snapshot.total_used, 3884449)

    def test_format_helpers_cover_special_values(self) -> None:
        self.assertEqual(format_expires_at(0), "不过期或未提供")
        self.assertEqual(format_model_limits({}), "未提供模型限制")
        self.assertEqual(
            format_model_limits({"gpt-4o": 10, "o3": "unlimited"}),
            "gpt-4o: 10\no3: unlimited",
        )
        self.assertEqual(
            format_fetched_at(datetime(2026, 3, 22, 10, 30, 45)),
            "2026-03-22 10:30:45",
        )


if __name__ == "__main__":
    unittest.main()
