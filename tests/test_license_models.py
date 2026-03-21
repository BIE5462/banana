from __future__ import annotations

import unittest

from license_models import LicenseOptions, LicenseState


class LicenseModelsTests(unittest.TestCase):
    def test_license_options_round_trip_and_validate(self) -> None:
        options = LicenseOptions(
            enabled=True,
            app_id="demo-app",
            app_key="demo-app-key",
            encrypt_key="1234567890abcdef",
            host_url="https://license.example.com",
            request_timeout_sec=9,
            bind_hardware=False,
            remember_card_default=False,
        )

        cloned = LicenseOptions.from_dict(options.to_dict())
        valid, message = cloned.validate()

        self.assertEqual(cloned, options)
        self.assertTrue(valid)
        self.assertEqual(message, "")

    def test_license_state_round_trip(self) -> None:
        state = LicenseState(
            remember_card=True,
            card_ciphertext="cipher",
            device_fingerprint="fp",
            fallback_device_id="fallback",
            last_login_at="2026-03-21T21:00:00",
            last_expire_at="2026-12-31 23:59:59",
            last_card_info={"token": "abc"},
        )

        cloned = LicenseState.from_dict(state.to_dict())

        self.assertEqual(cloned, state)


if __name__ == "__main__":
    unittest.main()
