from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYSIDE_DISABLE_INTERNAL_QT_CONF", "1")
sys.path = [entry for entry in sys.path if entry is not None]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from api_usage_service import ApiUsageSnapshot
from config import AppConfig, GenerationSettings
import main as main_module


class MainWindowApiUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self) -> main_module.MainWindow:
        with patch("main.ConfigManager.load", return_value=AppConfig.default()):
            window = main_module.MainWindow()
        self.addCleanup(window.close)
        return window

    def build_snapshot(self) -> ApiUsageSnapshot:
        return ApiUsageSnapshot(
            name="演示 Token",
            total_available=80,
            total_granted=100,
            total_used=20,
            expires_at=0,
            model_limits={"gpt-4o": 10},
            model_limits_enabled=True,
            unlimited_quota=False,
            fetched_at=datetime(2026, 3, 22, 10, 30, 45),
            base_url="https://wuaiapi.com",
        )

    def test_usage_card_shows_unconfigured_state_by_default(self) -> None:
        window = self.create_window()

        self.assertEqual(window.usage_value_label.text(), "--")
        self.assertEqual(window.usage_meta_label.text(), "请先填写 URL 和 API Key")
        self.assertEqual(window.usage_state_badge.property("usageState"), "inactive")
        self.assertTrue(window.usage_progress_bar.isHidden())

    def test_usage_card_renders_snapshot(self) -> None:
        window = self.create_window()
        window.api_url_edit.setText("https://wuaiapi.com")
        window.api_key_edit.setText("sk-test")
        window.usage_snapshot = self.build_snapshot()
        window.usage_error = ""
        window.usage_loading = False

        window.render_usage_state()

        self.assertEqual(window.usage_value_label.text(), "80")
        self.assertEqual(window.usage_meta_label.text(), "已使用 20 / 总授予 100")
        self.assertEqual(window.usage_state_badge.text(), "已同步")
        self.assertEqual(window.usage_state_badge.property("usageState"), "success")
        self.assertFalse(window.usage_progress_bar.isHidden())
        self.assertEqual(window.usage_progress_bar.value(), 200)
        self.assertIn("2026-03-22 10:30:45", window.usage_time_label.text())

    def test_usage_card_keeps_last_snapshot_when_refresh_fails(self) -> None:
        window = self.create_window()
        window.api_url_edit.setText("https://wuaiapi.com")
        window.api_key_edit.setText("sk-test")
        window.usage_snapshot = self.build_snapshot()
        window.usage_error = "额度查询失败：network down"
        window.usage_loading = False

        window.render_usage_state()

        self.assertEqual(window.usage_value_label.text(), "80")
        self.assertEqual(window.usage_state_badge.text(), "刷新失败")
        self.assertEqual(window.usage_state_badge.property("usageState"), "warning")
        self.assertFalse(window.usage_error_label.isHidden())
        self.assertIn("network down", window.usage_error_label.text())

    def test_load_config_triggers_auto_refresh_when_api_credentials_exist(self) -> None:
        config = AppConfig.default()
        config.api_base_url = "https://wuaiapi.com"
        config.api_key = "sk-test"

        with patch.object(
            main_module.MainWindow,
            "refresh_usage",
            autospec=True,
        ) as mocked_refresh, patch(
            "main.ConfigManager.load",
            return_value=config,
        ):
            window = main_module.MainWindow()
            self.addCleanup(window.close)

        mocked_refresh.assert_called_once_with(window, auto=True)

    def test_editing_finished_triggers_usage_refresh_handler(self) -> None:
        with patch.object(
            main_module.MainWindow,
            "handle_usage_input_finished",
            autospec=True,
        ) as mocked_handler, patch(
            "main.ConfigManager.load",
            return_value=AppConfig.default(),
        ):
            window = main_module.MainWindow()
            self.addCleanup(window.close)

        mocked_handler.reset_mock()
        window.api_url_edit.editingFinished.emit()
        window.api_key_edit.editingFinished.emit()

        self.assertEqual(mocked_handler.call_count, 2)

    def test_settings_dialog_shows_usage_details(self) -> None:
        snapshot = self.build_snapshot()
        dialog = main_module.SettingsDialog(
            api_base_url="https://wuaiapi.com",
            api_key="sk-test",
            generation_settings=GenerationSettings(),
            usage_snapshot=snapshot,
            usage_error="",
        )
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.usage_name_value.text(), "演示 Token")
        self.assertEqual(dialog.usage_available_value.text(), "80")
        self.assertEqual(dialog.usage_limits_enabled_value.text(), "已启用")
        self.assertIn("gpt-4o: 10", dialog.usage_model_limits_edit.toPlainText())
        self.assertEqual(dialog.usage_error_value.text(), "最近刷新成功")


if __name__ == "__main__":
    unittest.main()
