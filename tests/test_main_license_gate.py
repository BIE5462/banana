from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from license_models import LicenseOptions
from license_bootstrap import authorize_before_launch


class MainLicenseGateTests(unittest.TestCase):
    def test_invalid_license_config_exits_with_error(self) -> None:
        with patch(
            "license_bootstrap.load_license_options",
            return_value=(False, "授权配置加载失败", LicenseOptions()),
        ) as _mocked_load:
            mocked_error = MagicMock()
            exit_code, manager = authorize_before_launch(
                dialog_factory=MagicMock(),
                show_error=mocked_error,
                accepted_code=1,
            )

        self.assertEqual(exit_code, 1)
        self.assertIsNone(manager)
        mocked_error.assert_called_once()

    def test_disabled_license_skips_dialog(self) -> None:
        fake_manager = SimpleNamespace(is_enabled=False)
        with patch(
            "license_bootstrap.load_license_options",
            return_value=(True, "", LicenseOptions(enabled=False)),
        ), patch(
            "license_bootstrap.LicenseManager", return_value=fake_manager
        ):
            mocked_dialog_factory = MagicMock()
            exit_code, manager = authorize_before_launch(
                dialog_factory=mocked_dialog_factory,
                show_error=MagicMock(),
                accepted_code=1,
            )

        self.assertIsNone(exit_code)
        self.assertIs(manager, fake_manager)
        mocked_dialog_factory.assert_not_called()

    def test_enabled_license_cancel_returns_zero(self) -> None:
        fake_manager = SimpleNamespace(is_enabled=True)
        fake_dialog = MagicMock()
        fake_dialog.exec.return_value = 0
        with patch(
            "license_bootstrap.load_license_options",
            return_value=(True, "", LicenseOptions(enabled=True)),
        ), patch(
            "license_bootstrap.LicenseManager", return_value=fake_manager
        ):
            exit_code, manager = authorize_before_launch(
                dialog_factory=MagicMock(return_value=fake_dialog),
                show_error=MagicMock(),
                accepted_code=1,
            )

        self.assertEqual(exit_code, 0)
        self.assertIs(manager, fake_manager)


if __name__ == "__main__":
    unittest.main()
