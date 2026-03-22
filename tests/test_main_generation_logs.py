from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYSIDE_DISABLE_INTERNAL_QT_CONF", "1")
sys.path = [entry for entry in sys.path if entry is not None]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from config import AppConfig
from generation_log_service import GenerationLogEntry, GenerationLogItem
import main as main_module


class MainWindowGenerationLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self) -> main_module.MainWindow:
        with patch("main.ConfigManager.load", return_value=AppConfig.default()):
            window = main_module.MainWindow()
        self.addCleanup(window.close)
        return window

    def build_summary(self, *, status: str, stopped: bool = False) -> dict[str, object]:
        success_count = 1 if status in {"success", "partial", "stopped"} else 0
        failure_count = 1 if status in {"partial", "failed"} else 0
        return {
            "started_at": "2026-03-22T10:00:00+08:00",
            "finished_at": "2026-03-22T10:01:00+08:00",
            "status": status,
            "total": 2,
            "completed": 2,
            "success_count": success_count,
            "failure_count": failure_count,
            "stopped": stopped,
            "output_dir": "D:/output",
            "api_base_url": "https://example.com",
            "model_type": "gemini-2.5-flash-image",
            "temperature": 0.8,
            "top_p": 0.65,
            "aspect_ratio": "Auto",
            "image_size": "2K",
            "variants_per_group": 1,
            "items": [
                {
                    "group_index": 0,
                    "match_key": "look-1",
                    "prompt_text": "polished studio shot",
                    "variant_index": 0,
                    "success": success_count > 0,
                    "reference_images": ["D:/refs/demo.png"],
                    "saved_paths": ["D:/output/demo.png"] if success_count > 0 else [],
                    "error": "" if success_count > 0 else "request failed",
                    "elapsed_seconds": 1.5,
                    "request_seconds": 1.1,
                    "seed": 12,
                }
            ],
        }

    def test_view_logs_button_opens_dialog(self) -> None:
        window = self.create_window()

        with patch("main.load_generation_log_entries", return_value=[]), patch.object(
            main_module.GenerationLogDialog, "exec", autospec=True, return_value=0
        ) as mocked_exec:
            window.view_logs_button.click()

        mocked_exec.assert_called_once()

    def test_handle_finished_saves_expected_status_variants(self) -> None:
        window = self.create_window()

        cases = [
            ("success", False, "success"),
            ("partial", False, "partial"),
            ("failed", False, "failed"),
            ("stopped", True, "stopped"),
        ]
        with patch("main.append_generation_log_entry") as mocked_append, patch.object(
            main_module.QMessageBox, "information"
        ):
            for raw_status, stopped, expected_status in cases:
                with self.subTest(status=expected_status):
                    window.handle_finished(self.build_summary(status=raw_status, stopped=stopped))
                    saved_entry = mocked_append.call_args[0][0]
                    self.assertEqual(saved_entry.status, expected_status)
                    self.assertEqual(saved_entry.output_dir, "D:/output")

    def test_generation_log_dialog_renders_latest_entry_and_preview(self) -> None:
        window = self.create_window()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            image_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6l8AAAAASUVORK5CYII="
                )
            )
            entry = GenerationLogEntry(
                id="entry-1",
                started_at="2026-03-22T10:00:00+08:00",
                finished_at="2026-03-22T10:01:00+08:00",
                status="success",
                total_planned=1,
                completed=1,
                success_count=1,
                failure_count=0,
                stopped=False,
                output_dir=temp_dir,
                api_base_url="https://example.com",
                model_type="gemini-2.5-flash-image",
                temperature=0.8,
                top_p=0.65,
                aspect_ratio="Auto",
                image_size="2K",
                variants_per_group=1,
                items=[
                    GenerationLogItem(
                        group_index=0,
                        match_key="look-1",
                        prompt_text="studio lighting",
                        variant_index=0,
                        success=True,
                        reference_images=["D:/refs/look.png"],
                        saved_paths=[str(image_path)],
                        elapsed_seconds=1.1,
                        request_seconds=0.8,
                        seed=9,
                    )
                ],
            )

            with patch("main.load_generation_log_entries", return_value=[entry]):
                dialog = main_module.GenerationLogDialog(window)
                self.addCleanup(dialog.close)
                self.app.processEvents()

        self.assertEqual(dialog.history_list.count(), 1)
        self.assertEqual(dialog.summary_status_badge.text(), "成功")
        self.assertEqual(dialog.item_table.rowCount(), 1)
        preview = dialog.log_preview_label.pixmap()
        self.assertIsNotNone(preview)
        self.assertFalse(preview.isNull())

    def test_generation_log_dialog_shows_missing_file_message(self) -> None:
        window = self.create_window()
        entry = GenerationLogEntry(
            id="entry-2",
            started_at="2026-03-22T10:00:00+08:00",
            finished_at="2026-03-22T10:01:00+08:00",
            status="success",
            total_planned=1,
            completed=1,
            success_count=1,
            failure_count=0,
            stopped=False,
            output_dir="D:/missing-output",
            api_base_url="https://example.com",
            model_type="gemini-2.5-flash-image",
            temperature=0.8,
            top_p=0.65,
            aspect_ratio="Auto",
            image_size="2K",
            variants_per_group=1,
            items=[
                GenerationLogItem(
                    group_index=0,
                    match_key="look-2",
                    prompt_text="missing image",
                    variant_index=0,
                    success=True,
                    reference_images=[],
                    saved_paths=["D:/missing-output/not-found.png"],
                    elapsed_seconds=1.0,
                    request_seconds=0.7,
                    seed=None,
                )
            ],
        )

        with patch("main.load_generation_log_entries", return_value=[entry]):
            dialog = main_module.GenerationLogDialog(window)
            self.addCleanup(dialog.close)
            self.app.processEvents()

        self.assertIn("图片文件不存在或无法加载", dialog.log_preview_label.text())


if __name__ == "__main__":
    unittest.main()
