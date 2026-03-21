from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class CallableWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class LicenseLoginDialog(QDialog):
    def __init__(
        self,
        license_manager: Any,
        parent: QWidget | None = None,
        worker_factory: Callable[..., CallableWorker] | None = None,
    ) -> None:
        super().__init__(parent)
        self.license_manager = license_manager
        self.worker_factory = worker_factory or CallableWorker
        self.worker: CallableWorker | None = None
        self.setWindowTitle("授权登录")
        self.setModal(True)
        self.setMinimumSize(520, 340)
        self._build_ui()
        self._apply_styles()
        self._load_default_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        panel = QFrame()
        panel.setObjectName("panelCard")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(14)

        title_label = QLabel("卡密验证")
        title_label.setObjectName("titleLabel")
        hint_label = QLabel("请输入有效卡密后进入系统。")
        hint_label.setObjectName("hintLabel")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(12)

        self.card_edit = QLineEdit()
        self.card_edit.setPlaceholderText("请输入卡密")
        self.card_edit.setClearButtonEnabled(True)

        self.remember_check = QCheckBox("记住卡密")
        self.remember_check.setCursor(Qt.CursorShape.PointingHandCursor)

        form.addRow("卡密", self.card_edit)
        form.addRow("", self.remember_check)

        self.status_label = QLabel("请输入卡密后登录")
        self.status_label.setObjectName("statusInfo")
        self.status_label.setWordWrap(True)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.login_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.login_button.setText("登录")
        self.cancel_button.setText("取消")
        self.login_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)

        panel_layout.addWidget(title_label)
        panel_layout.addWidget(hint_label)
        panel_layout.addLayout(form)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.button_box)

        layout.addWidget(panel)

        self.card_edit.returnPressed.connect(self._handle_login)
        self.button_box.accepted.connect(self._handle_login)
        self.button_box.rejected.connect(self.reject)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #f6f4ef;
                color: #171717;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 13px;
            }
            #panelCard {
                background: #fffdf9;
                border: 1px solid #d7d2c6;
                border-radius: 14px;
            }
            #titleLabel {
                font-size: 24px;
                font-weight: 700;
            }
            #hintLabel {
                color: #5f5b52;
            }
            #statusInfo {
                min-height: 22px;
                color: #5f5b52;
            }
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d5d0c3;
                border-radius: 8px;
                padding: 7px 9px;
            }
            QLineEdit:focus {
                border: 1px solid #d4af37;
            }
            QPushButton {
                background: #171717;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #2d2d2d;
            }
            QPushButton:disabled {
                background: #a6a6a6;
                color: #f2f2f2;
            }
            """
        )

    def _load_default_values(self) -> None:
        remembered_card = self.license_manager.get_remembered_card()
        self.remember_check.setChecked(self.license_manager.options.remember_card_default)
        if remembered_card:
            self.card_edit.setText(remembered_card)
            self._set_status_message("已检测到保存的卡密，可直接登录。", "info")

    def _set_loading(self, loading: bool) -> None:
        self.card_edit.setEnabled(not loading)
        self.remember_check.setEnabled(not loading)
        self.login_button.setEnabled(not loading)
        self.cancel_button.setEnabled(not loading)
        self.login_button.setText("登录中..." if loading else "登录")

    def _set_status_message(self, message: str, state: str) -> None:
        self.status_label.setText(message)
        colors = {
            "info": "#5f5b52",
            "success": "#166534",
            "warning": "#92400e",
            "error": "#b42318",
        }
        self.status_label.setStyleSheet(f"color: {colors.get(state, '#5f5b52')};")

    def _handle_login(self) -> None:
        card = self.card_edit.text().strip()
        if not card:
            self._set_status_message("请输入卡密后再登录。", "warning")
            self.card_edit.setFocus()
            return

        self._set_loading(True)
        self._set_status_message("正在验证卡密，请稍候...", "info")
        self.worker = self.worker_factory(
            self.license_manager.login,
            card,
            self.remember_check.isChecked(),
        )
        self.worker.succeeded.connect(self._on_login_finished)
        self.worker.failed.connect(self._on_login_failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _cleanup_worker(self) -> None:
        self.worker = None

    def _on_login_finished(self, result: object) -> None:
        self._set_loading(False)
        success, message, _ = result
        if success:
            self._set_status_message(message or "登录成功", "success")
            self.accept()
            return
        self._set_status_message(message or "登录失败，请检查卡密。", "error")
        QMessageBox.warning(self, "登录失败", message or "登录失败，请检查卡密。")

    def _on_login_failed(self, message: str) -> None:
        self._set_loading(False)
        self._set_status_message(message, "error")
        QMessageBox.critical(self, "登录异常", message)
