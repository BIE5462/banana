from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, QSize
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSplitter,
)

from GeminiImage import (
    BatchTask,
    GeminiImageGenerator,
    build_batch_tasks,
    list_image_files,
)
from config import AppConfig, ConfigManager, FolderSlot, GenerationSettings

WINDOW_TITLE = "NanoBanana Batch"


def create_line_button_row(
    line_edit: QLineEdit, button_text: str
) -> tuple[QWidget, QPushButton]:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(line_edit)
    button = QPushButton(button_text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    layout.addWidget(button)
    return container, button


class FolderSlotWidget(QFrame):
    changed = Signal()
    remove_requested = Signal(object)

    def __init__(
        self, slot: FolderSlot | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("slotCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.enable_checkbox = QCheckBox("启用")
        self.enable_checkbox.setChecked(True)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("槽位名称，例如：饰品")

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择参考图片文件夹")

        self.count_label = QLabel("0 张图片")
        self.count_label.setObjectName("mutedLabel")
        self.count_label.setMinimumWidth(90)

        self.browse_button = QPushButton("选择文件夹")
        self.remove_button = QPushButton("删除")
        for button in (self.browse_button, self.remove_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self.enable_checkbox)
        top_row.addWidget(QLabel("名称"))
        top_row.addWidget(self.name_edit, 1)
        top_row.addWidget(self.count_label)
        top_row.addWidget(self.remove_button)

        path_row = QHBoxLayout()
        path_row.setSpacing(10)
        path_row.addWidget(QLabel("目录"))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addLayout(top_row)
        layout.addLayout(path_row)

        self.browse_button.clicked.connect(self.choose_directory)
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.enable_checkbox.toggled.connect(self._emit_changed)
        self.name_edit.textChanged.connect(self._emit_changed)
        self.path_edit.textChanged.connect(self._handle_path_changed)

        if slot is not None:
            self.set_slot(slot)
        else:
            self.refresh_count()

    def _emit_changed(self) -> None:
        self.changed.emit()

    def _handle_path_changed(self) -> None:
        self.refresh_count()
        self.changed.emit()

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择参考图片文件夹",
            self.path_edit.text().strip() or str(Path.home()),
        )
        if directory:
            self.path_edit.setText(directory)

    def set_slot(self, slot: FolderSlot) -> None:
        self.name_edit.setText(slot.name)
        self.path_edit.setText(slot.path)
        self.enable_checkbox.setChecked(slot.enabled)
        self.refresh_count()

    def to_slot(self) -> FolderSlot:
        name = self.name_edit.text().strip() or "参考图"
        return FolderSlot(
            name=name,
            path=self.path_edit.text().strip(),
            enabled=self.enable_checkbox.isChecked(),
        )

    def refresh_count(self) -> int:
        count = len(list_image_files(self.path_edit.text().strip()))
        self.count_label.setText(f"{count} 张图片")
        return count


class BatchWorker(QObject):
    log_message = Signal(str)
    progress_changed = Signal(int, int)
    preview_ready = Signal(str)
    finished = Signal(object)

    def __init__(self, config: AppConfig, tasks: list[BatchTask]) -> None:
        super().__init__()
        self.config = config
        self.tasks = tasks
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        settings = self.config.generation_settings
        generator = GeminiImageGenerator(
            api_base_url=self.config.api_base_url,
            api_key=self.config.api_key,
            model_type=settings.model_type,
        )

        total = len(self.tasks) * max(1, settings.variants_per_group)
        completed = 0
        success_count = 0
        failure_count = 0
        stopped = False

        self.log_message.emit(
            f"任务开始，共 {len(self.tasks)} 组，计划生成 {total} 次。"
        )

        for task in self.tasks:
            if self._stop_requested:
                stopped = True
                break

            self.log_message.emit(
                f"开始第 {task.group_index + 1}/{len(self.tasks)} 组，参考图 {len(task.reference_images)} 张。"
            )

            for variant_index in range(max(1, settings.variants_per_group)):
                if self._stop_requested:
                    stopped = True
                    break

                seed = None
                if settings.seed_enabled:
                    seed = (
                        settings.base_seed
                        + task.group_index * max(1, settings.variants_per_group)
                        + variant_index
                    )

                result = generator.generate_single_image(
                    task=task,
                    prompt=settings.prompt,
                    output_dir=self.config.output_dir,
                    variant_index=variant_index,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    aspect_ratio=settings.aspect_ratio,
                    timeout=settings.timeout,
                    seed=seed,
                )

                completed += 1
                self.progress_changed.emit(completed, total)
                if result.success:
                    success_count += 1
                    image_count = len(result.saved_paths)
                    self.log_message.emit(
                        f"第 {task.group_index + 1} 组 / 变体 {variant_index + 1} 成功，"
                        f"保存 {image_count} 张，耗时 {result.elapsed_seconds:.1f}s。"
                    )
                    if result.saved_paths:
                        self.preview_ready.emit(result.saved_paths[-1])
                else:
                    failure_count += 1
                    self.log_message.emit(
                        f"第 {task.group_index + 1} 组 / 变体 {variant_index + 1} 失败：{result.error}"
                    )

            if stopped:
                break

        summary = {
            "total": total,
            "completed": completed,
            "success_count": success_count,
            "failure_count": failure_count,
            "stopped": stopped,
            "output_dir": self.config.output_dir,
        }
        self.finished.emit(summary)


class SettingsDialog(QDialog):
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        generation_settings: GenerationSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("高级设置")
        self.setModal(True)
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

        self.api_url_edit = QLineEdit(api_base_url)
        self.api_url_edit.setPlaceholderText(
            "https://generativelanguage.googleapis.com"
        )

        self.api_key_edit = QLineEdit(api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("输入 API Key")

        self.model_edit = QLineEdit(generation_settings.model_type)
        self.model_edit.setPlaceholderText("gemini-2.5-flash-image")

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(generation_settings.temperature)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(generation_settings.top_p)

        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["Auto", "1:1", "3:4", "4:3", "9:16", "16:9"])
        self.aspect_ratio_combo.setCurrentText(generation_settings.aspect_ratio)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setSingleStep(10)
        self.timeout_spin.setSuffix(" 秒")
        self.timeout_spin.setValue(generation_settings.timeout)

        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(1, 20)
        self.variants_spin.setValue(generation_settings.variants_per_group)

        self.seed_enabled_check = QCheckBox("启用固定 Seed")
        self.seed_enabled_check.setChecked(generation_settings.seed_enabled)
        self.seed_enabled_check.setCursor(Qt.CursorShape.PointingHandCursor)

        self.base_seed_spin = QSpinBox()
        self.base_seed_spin.setRange(0, 2_147_483_647)
        self.base_seed_spin.setValue(generation_settings.base_seed)

        form.addRow("URL", self.api_url_edit)
        form.addRow("API Key", self.api_key_edit)
        form.addRow("模型", self.model_edit)
        form.addRow("Temperature", self.temperature_spin)
        form.addRow("Top P", self.top_p_spin)
        form.addRow("比例", self.aspect_ratio_combo)
        form.addRow("超时", self.timeout_spin)
        form.addRow("每组生成数量", self.variants_spin)
        form.addRow(self.seed_enabled_check, self.base_seed_spin)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addLayout(form)
        layout.addStretch(1)
        layout.addWidget(button_box)

    def get_values(self) -> tuple[str, str, GenerationSettings]:
        return (
            self.api_url_edit.text().strip(),
            self.api_key_edit.text().strip(),
            GenerationSettings(
                prompt="",
                model_type=self.model_edit.text().strip(),
                temperature=self.temperature_spin.value(),
                top_p=self.top_p_spin.value(),
                aspect_ratio=self.aspect_ratio_combo.currentText(),
                timeout=self.timeout_spin.value(),
                variants_per_group=self.variants_spin.value(),
                seed_enabled=self.seed_enabled_check.isChecked(),
                base_seed=self.base_seed_spin.value(),
            ),
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1000, 800)

        self.thread: QThread | None = None
        self.worker: BatchWorker | None = None
        self.slot_widgets: list[FolderSlotWidget] = []
        self.generation_settings = GenerationSettings()

        self._build_ui()
        self._apply_styles()
        self.load_config()
        self.update_task_summary()

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)

        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        title_label = QLabel("NanoBanana Batch")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("按参考槽位顺序批量组图，调用 API 生成并保存结果。")
        subtitle_label.setObjectName("mutedLabel")

        root_layout.addWidget(title_label)
        root_layout.addWidget(subtitle_label)

        three_panel_layout = QHBoxLayout()
        three_panel_layout.setSpacing(14)
        three_panel_layout.addWidget(self._build_slots_group(), 4)
        three_panel_layout.addWidget(self._build_basic_config_group(), 3)
        three_panel_layout.addWidget(self._build_advanced_settings_group(), 2)
        root_layout.addLayout(three_panel_layout)

        root_layout.addWidget(self._build_action_group())
        root_layout.addWidget(self._build_result_group(), 1)

        self.statusBar().showMessage("就绪")

    def _build_slots_group(self) -> QGroupBox:
        group = QGroupBox("参考文件夹")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        self.add_slot_button = QPushButton("新增槽位")
        self.add_slot_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.task_summary_label = QLabel("已启用 0 个槽位，可执行 0 组")
        self.task_summary_label.setObjectName("summaryLabel")
        toolbar.addWidget(self.add_slot_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.task_summary_label)
        layout.addLayout(toolbar)

        self.slot_container = QWidget()
        self.slot_layout = QVBoxLayout(self.slot_container)
        self.slot_layout.setContentsMargins(0, 0, 0, 0)
        self.slot_layout.setSpacing(10)
        self.slot_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(self.slot_container)
        layout.addWidget(scroll_area)

        self.add_slot_button.clicked.connect(lambda: self.add_slot())
        return group

    def _build_basic_config_group(self) -> QGroupBox:
        group = QGroupBox("基础配置")
        basic_form = QFormLayout(group)
        basic_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        basic_form.setSpacing(10)

        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText(
            "https://generativelanguage.googleapis.com"
        )

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("输入 API Key")

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("输入统一的提示词")
        self.prompt_edit.setFixedHeight(110)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录")
        output_row, self.output_dir_button = create_line_button_row(
            self.output_dir_edit,
            "选择目录",
        )

        basic_form.addRow("URL", self.api_url_edit)
        basic_form.addRow("API Key", self.api_key_edit)
        basic_form.addRow("Prompt", self.prompt_edit)
        basic_form.addRow("输出目录", output_row)

        self.output_dir_button.clicked.connect(self.choose_output_directory)
        return group

    def _build_advanced_settings_group(self) -> QGroupBox:
        group = QGroupBox("高级设置")
        settings_layout = QVBoxLayout(group)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        self.settings_summary_label = QLabel()
        self.settings_summary_label.setWordWrap(True)
        self.settings_summary_label.setObjectName("mutedLabel")

        self.settings_button = QPushButton("设置")
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)

        settings_layout.addWidget(self.settings_summary_label)
        settings_layout.addStretch(1)
        settings_layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.settings_button.clicked.connect(self.open_settings_dialog)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("任务控制")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        self.start_button = QPushButton("开始批量生成")
        self.stop_button = QPushButton("安全停止")
        self.save_button = QPushButton("保存配置")
        for button in (self.start_button, self.stop_button, self.save_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.progress_bar, 1)

        self.start_button.clicked.connect(self.start_batch)
        self.stop_button.clicked.connect(self.stop_batch)
        self.save_button.clicked.connect(self.save_config)
        return group

    def _build_result_group(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)

        preview_group = QGroupBox("最新结果预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("暂无预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setMinimumSize(QSize(360, 360))
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_layout.addWidget(self.preview_label)

        splitter.addWidget(log_group)
        splitter.addWidget(preview_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f6f4ef;
                color: #171717;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #d7d2c6;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 16px;
                background: #fffdf9;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            #titleLabel {
                font-size: 28px;
                font-weight: 700;
            }
            #mutedLabel {
                color: #5f5b52;
            }
            #summaryLabel {
                color: #7a5f14;
                font-weight: 600;
            }
            #slotCard {
                background: #ffffff;
                border: 1px solid #e4ded0;
                border-radius: 12px;
            }
            #previewLabel {
                background: #f0ede6;
                border: 1px dashed #c4bba7;
                border-radius: 12px;
                color: #5f5b52;
            }
            QPushButton, QToolButton {
                background: #171717;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #2d2d2d;
            }
            QPushButton:disabled {
                background: #a6a6a6;
                color: #f2f2f2;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #d5d0c3;
                border-radius: 8px;
                padding: 7px 9px;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #d4af37;
            }
            QCheckBox {
                spacing: 8px;
            }
            QProgressBar {
                background: #ece7db;
                border: 1px solid #d5d0c3;
                border-radius: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #d4af37;
                border-radius: 7px;
            }
            """
        )

    def add_slot(self, slot: FolderSlot | None = None) -> None:
        widget = FolderSlotWidget(slot)
        widget.changed.connect(self.update_task_summary)
        widget.remove_requested.connect(self.remove_slot)
        self.slot_widgets.append(widget)
        self.slot_layout.insertWidget(max(0, self.slot_layout.count() - 1), widget)
        self.update_task_summary()

    def remove_slot(self, widget: FolderSlotWidget) -> None:
        if len(self.slot_widgets) <= 1:
            QMessageBox.warning(self, "无法删除", "至少保留一个参考槽位。")
            return

        self.slot_widgets.remove(widget)
        widget.setParent(None)
        widget.deleteLater()
        self.update_task_summary()

    def choose_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.output_dir_edit.text().strip() or str(Path.home()),
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def load_config(self) -> None:
        config = ConfigManager.load()

        self.api_url_edit.setText(config.api_base_url)
        self.api_key_edit.setText(config.api_key)
        self.output_dir_edit.setText(config.output_dir)

        self.generation_settings = config.generation_settings
        self.prompt_edit.setPlainText(self.generation_settings.prompt)

        self.slot_widgets = []
        while self.slot_layout.count() > 1:
            item = self.slot_layout.takeAt(0)
            child = item.widget()
            if child:
                child.deleteLater()

        for slot in config.folder_slots:
            self.add_slot(slot)

        if not self.slot_widgets:
            self.add_slot(FolderSlot(name="参考图"))
        self.update_settings_summary()

    def collect_config(self) -> AppConfig:
        settings = GenerationSettings(
            prompt=self.prompt_edit.toPlainText().strip(),
            model_type=self.generation_settings.model_type,
            temperature=self.generation_settings.temperature,
            top_p=self.generation_settings.top_p,
            aspect_ratio=self.generation_settings.aspect_ratio,
            timeout=self.generation_settings.timeout,
            variants_per_group=self.generation_settings.variants_per_group,
            seed_enabled=self.generation_settings.seed_enabled,
            base_seed=self.generation_settings.base_seed,
        )
        return AppConfig(
            api_base_url=self.api_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            output_dir=self.output_dir_edit.text().strip(),
            folder_slots=[widget.to_slot() for widget in self.slot_widgets],
            generation_settings=settings,
        )

    def update_settings_summary(self) -> None:
        api_key = self.api_key_edit.text().strip()
        api_base_url = self.api_url_edit.text().strip()

        masked_key = "未配置"
        if api_key:
            if len(api_key) <= 8:
                masked_key = "*" * len(api_key)
            else:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}"

        settings = self.generation_settings
        self.settings_summary_label.setText(
            "URL：{url}\n"
            "API Key：{key}\n"
            "模型：{model}\n"
            "Temperature：{temperature:.2f}   Top P：{top_p:.2f}\n"
            "比例：{ratio}   超时：{timeout} 秒   每组：{variants} 次\n"
            "Seed：{seed}".format(
                url=api_base_url or "未配置",
                key=masked_key,
                model=settings.model_type or "未配置",
                temperature=settings.temperature,
                top_p=settings.top_p,
                ratio=settings.aspect_ratio,
                timeout=settings.timeout,
                variants=settings.variants_per_group,
                seed=str(settings.base_seed) if settings.seed_enabled else "随机",
            )
        )

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(
            api_base_url=self.api_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            generation_settings=self.generation_settings,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        api_base_url, api_key, generation_settings = dialog.get_values()
        generation_settings.prompt = self.prompt_edit.toPlainText().strip()
        self.api_url_edit.setText(api_base_url)
        self.api_key_edit.setText(api_key)
        self.generation_settings = generation_settings
        self.update_settings_summary()

    def update_task_summary(self) -> None:
        enabled_counts: list[int] = []
        enabled_slots = 0
        for widget in self.slot_widgets:
            slot = widget.to_slot()
            count = widget.refresh_count()
            if slot.enabled and slot.path:
                enabled_slots += 1
                enabled_counts.append(count)

        executable_groups = min(enabled_counts) if enabled_counts else 0
        self.task_summary_label.setText(
            f"已启用 {enabled_slots} 个槽位，可执行 {executable_groups} 组"
        )

    def save_config(self) -> None:
        try:
            saved_path = ConfigManager.save(self.collect_config())
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        self.statusBar().showMessage(f"配置已保存到 {saved_path}", 4000)
        self.append_log(f"配置已保存：{saved_path}")

    def validate_and_build_tasks(self) -> tuple[AppConfig, list[BatchTask]]:
        config_data = self.collect_config()
        settings = config_data.generation_settings

        if not config_data.api_base_url:
            raise ValueError("请填写 API URL。")
        if not config_data.api_key:
            raise ValueError("请填写 API Key。")
        if not settings.prompt:
            raise ValueError("请填写 Prompt。")
        if not config_data.output_dir:
            raise ValueError("请选择输出目录。")

        enabled_slot_sources: list[tuple[str, list[str]]] = []
        for slot in config_data.folder_slots:
            if not slot.enabled:
                continue
            if not slot.path:
                raise ValueError(f"槽位“{slot.name}”未选择文件夹。")

            folder_path = Path(slot.path)
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"槽位“{slot.name}”的目录不存在。")

            image_files = list_image_files(folder_path)
            if not image_files:
                raise ValueError(f"槽位“{slot.name}”目录中没有可用图片。")

            enabled_slot_sources.append((slot.name, image_files))

        if not enabled_slot_sources:
            raise ValueError("至少启用一个有效的参考槽位。")

        tasks = build_batch_tasks(enabled_slot_sources)
        if not tasks:
            raise ValueError("当前配置无法组成任何批处理任务。")

        return config_data, tasks

    def set_running_state(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.save_button.setEnabled(not running)
        self.add_slot_button.setEnabled(not running)

        for widget in self.slot_widgets:
            widget.setEnabled(not running)

        for control in (
            self.api_url_edit,
            self.api_key_edit,
            self.output_dir_edit,
            self.output_dir_button,
            self.prompt_edit,
            self.settings_button,
        ):
            control.setEnabled(not running)

    def start_batch(self) -> None:
        try:
            config_data, tasks = self.validate_and_build_tasks()
            Path(config_data.output_dir).mkdir(parents=True, exist_ok=True)
            ConfigManager.save(config_data)
        except Exception as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return

        self.log_edit.clear()
        self.preview_label.setText("等待生成结果")
        self.preview_label.setPixmap(QPixmap())
        self.progress_bar.setRange(
            0, len(tasks) * max(1, config_data.generation_settings.variants_per_group)
        )
        self.progress_bar.setValue(0)

        self.thread = QThread(self)
        self.worker = BatchWorker(config_data, tasks)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.append_log)
        self.worker.preview_ready.connect(self.update_preview)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.finished.connect(self.handle_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)

        self.set_running_state(True)
        self.statusBar().showMessage("任务执行中")
        self.thread.start()

    def stop_batch(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.append_log("已请求安全停止，将在当前请求完成后结束。")
        self.statusBar().showMessage("停止中")

    def append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)
        self.statusBar().showMessage(message, 5000)

    def update_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)

    def update_preview(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.preview_label.setText(f"无法预览：{image_path}")
            return

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def handle_finished(self, summary: object) -> None:
        data = dict(summary)
        stopped = data.get("stopped", False)
        completed = data.get("completed", 0)
        success_count = data.get("success_count", 0)
        failure_count = data.get("failure_count", 0)
        output_dir = data.get("output_dir", "")

        self.set_running_state(False)
        self.thread = None
        self.worker = None

        if stopped:
            self.append_log(
                f"任务已停止：完成 {completed} 次，成功 {success_count} 次，失败 {failure_count} 次。"
            )
            QMessageBox.information(
                self,
                "任务已停止",
                f"已完成 {completed} 次。\n成功 {success_count} 次，失败 {failure_count} 次。",
            )
        else:
            self.append_log(
                f"任务完成：成功 {success_count} 次，失败 {failure_count} 次，输出目录：{output_dir}"
            )
            QMessageBox.information(
                self,
                "任务完成",
                f"任务完成。\n成功 {success_count} 次，失败 {failure_count} 次。\n输出目录：{output_dir}",
            )

        self.statusBar().showMessage("已完成")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            answer = QMessageBox.question(
                self,
                "确认退出",
                "任务仍在执行中，退出会中断界面。是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        try:
            ConfigManager.save(self.collect_config())
        except Exception:
            pass
        event.accept()


def main() -> int:
    EXPIRE_DATE = datetime(2026, 3, 23)
    if datetime.now() > EXPIRE_DATE:
        app = QApplication(sys.argv)
        QMessageBox.critical(
            None, "软件已过期", "软件已于 2026年3月23日 过期，请联系开发者获取新版本。"
        )
        return 1

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
