from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QDialog

from license_login_dialog import LicenseLoginDialog


class _FakeOptions:
    remember_card_default = True


class _FakeManager:
    def __init__(self, result: tuple[bool, str, dict | None], remembered_card: str = ""):
        self.options = _FakeOptions()
        self._result = result
        self._remembered_card = remembered_card
        self.calls: list[tuple[str, bool]] = []

    def get_remembered_card(self) -> str:
        return self._remembered_card

    def login(self, card: str, remember_card: bool) -> tuple[bool, str, dict | None]:
        self.calls.append((card, remember_card))
        return self._result


class _ImmediateWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def start(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ManualLicenseLoginDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_prefills_remembered_card(self) -> None:
        dialog = LicenseLoginDialog(
            _FakeManager((True, "登录成功", {}), remembered_card="CARD-001"),
            worker_factory=_ImmediateWorker,
        )

        self.assertEqual(dialog.card_edit.text(), "CARD-001")

    def test_empty_card_shows_warning(self) -> None:
        dialog = LicenseLoginDialog(
            _FakeManager((True, "登录成功", {})),
            worker_factory=_ImmediateWorker,
        )
        dialog.card_edit.setText("")
        dialog._handle_login()

        self.assertIn("请输入卡密", dialog.status_label.text())

    def test_successful_login_accepts_dialog(self) -> None:
        manager = _FakeManager((True, "登录成功", {"token": "ok"}))
        dialog = LicenseLoginDialog(manager, worker_factory=_ImmediateWorker)
        dialog.card_edit.setText("CARD-002")
        dialog._handle_login()

        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(manager.calls, [("CARD-002", True)])

    def test_failed_login_keeps_dialog_open(self) -> None:
        manager = _FakeManager((False, "卡密不存在", None))
        dialog = LicenseLoginDialog(manager, worker_factory=_ImmediateWorker)
        dialog.card_edit.setText("CARD-404")

        with patch("license_login_dialog.QMessageBox.warning") as mocked_warning:
            dialog._handle_login()

        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertIn("卡密不存在", dialog.status_label.text())
        mocked_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
