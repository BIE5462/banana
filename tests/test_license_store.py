from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from license_models import LicenseState
from license_store import (
    EMBEDDED_LICENSE_OPTIONS,
    load_license_options,
    load_license_state,
    save_license_state,
)


class LicenseStoreTests(unittest.TestCase):
    def test_load_license_options_uses_embedded_values(self) -> None:
        success, _, options = load_license_options()

        self.assertTrue(success)
        self.assertTrue(options.enabled)
        self.assertEqual(options.app_id, EMBEDDED_LICENSE_OPTIONS.app_id)
        self.assertEqual(options.app_key, EMBEDDED_LICENSE_OPTIONS.app_key)
        self.assertEqual(options.encrypt_key, EMBEDDED_LICENSE_OPTIONS.encrypt_key)
        self.assertEqual(options.host_url, EMBEDDED_LICENSE_OPTIONS.host_url)

    def test_license_state_round_trip(self) -> None:
        state = LicenseState(
            remember_card=True,
            card_ciphertext="cipher",
            device_fingerprint="fingerprint",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "license_state.json"
            save_license_state(state, path)
            loaded = load_license_state(path)

        self.assertEqual(loaded.remember_card, state.remember_card)
        self.assertEqual(loaded.card_ciphertext, state.card_ciphertext)
        self.assertEqual(loaded.device_fingerprint, state.device_fingerprint)


if __name__ == "__main__":
    unittest.main()
