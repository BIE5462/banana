from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, Qt, QThread, Signal, QSize
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSplitter,
)

from GeminiImage import (
    BatchTask,
    GeminiImageGenerator,
    build_image_name_map,
    build_batch_tasks,
    list_image_files,
    parse_prompt_text_file,
    sort_match_keys,
)
from api_usage_service import (
    ApiUsageError,
    ApiUsageSnapshot,
    fetch_token_usage,
    format_expires_at,
    format_fetched_at,
    format_model_limits,
)
from config import AppConfig, ConfigManager, FolderSlot, GenerationSettings
from generation_log_service import (
    GenerationLogEntry,
    GenerationLogItem,
    append_entry as append_generation_log_entry,
    load_entries as load_generation_log_entries,
)
from license_bootstrap import authorize_before_launch as run_license_authorization
from license_login_dialog import LicenseLoginDialog
from license_service import LicenseManager

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


def format_quota_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def format_log_datetime(value: str) -> str:
    if not value:
        return "--"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def summarize_output_dir(path: str) -> str:
    if not path:
        return "未设置输出目录"
    normalized = Path(path)
    if normalized.name:
        return normalized.name
    return path


def format_generation_status(status: str) -> str:
    mapping = {
        "success": "成功",
        "partial": "部分成功",
        "failed": "失败",
        "stopped": "已停止",
    }
    return mapping.get(status, status or "未知")


def compute_generation_status(
    success_count: int, failure_count: int, stopped: bool
) -> str:
    if stopped:
        return "stopped"
    if success_count > 0 and failure_count == 0:
        return "success"
    if success_count > 0:
        return "partial"
    return "failed"


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
        self._started_at = datetime.now().astimezone().isoformat(timespec="seconds")

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
        log_items: list[dict[str, object]] = []

        self.log_message.emit(
            f"任务开始，共 {len(self.tasks)} 组，计划生成 {total} 次。"
        )

        for task in self.tasks:
            if self._stop_requested:
                stopped = True
                break

            self.log_message.emit(
                f"开始第 {task.group_index + 1}/{len(self.tasks)} 组 [{task.match_key}]，"
                f"参考图 {len(task.reference_images)} 张。"
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
                    prompt=task.prompt_text,
                    output_dir=self.config.output_dir,
                    variant_index=variant_index,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    aspect_ratio=settings.aspect_ratio,
                    image_size=settings.image_size,
                    timeout=settings.timeout,
                    seed=seed,
                )

                log_items.append(
                    {
                        "group_index": task.group_index,
                        "match_key": task.match_key,
                        "prompt_text": task.prompt_text,
                        "variant_index": result.variant_index,
                        "success": result.success,
                        "reference_images": list(result.reference_images),
                        "saved_paths": list(result.saved_paths),
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                        "request_seconds": result.request_seconds,
                        "seed": result.seed,
                    }
                )
                completed += 1
                self.progress_changed.emit(completed, total)
                if result.success:
                    success_count += 1
                    image_count = len(result.saved_paths)
                    self.log_message.emit(
                        f"第 {task.group_index + 1} 组 [{task.match_key}] / 变体 {variant_index + 1} 成功，"
                        f"保存 {image_count} 张，耗时 {result.elapsed_seconds:.1f}s。"
                    )
                    if result.saved_paths:
                        self.preview_ready.emit(result.saved_paths[-1])
                else:
                    failure_count += 1
                    self.log_message.emit(
                        f"第 {task.group_index + 1} 组 [{task.match_key}] / "
                        f"变体 {variant_index + 1} 失败：{result.error}"
                    )

            if stopped:
                break

        summary = {
            "started_at": self._started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": compute_generation_status(success_count, failure_count, stopped),
            "total": total,
            "completed": completed,
            "success_count": success_count,
            "failure_count": failure_count,
            "stopped": stopped,
            "output_dir": self.config.output_dir,
            "api_base_url": self.config.api_base_url,
            "model_type": settings.model_type,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "aspect_ratio": settings.aspect_ratio,
            "image_size": settings.image_size,
            "variants_per_group": settings.variants_per_group,
            "items": log_items,
        }
        self.finished.emit(summary)


class ApiUsageWorker(QObject):
    usage_loaded = Signal(object)
    usage_failed = Signal(str)
    finished = Signal()

    def __init__(self, base_url: str, api_key: str, timeout: int = 10) -> None:
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def run(self) -> None:
        try:
            snapshot = fetch_token_usage(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        except ApiUsageError as exc:
            self.usage_failed.emit(str(exc))
        except Exception as exc:
            self.usage_failed.emit(f"额度查询失败：{exc}")
        else:
            self.usage_loaded.emit(snapshot)
        finally:
            self.finished.emit()


class SettingsDialog(QDialog):
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        generation_settings: GenerationSettings,
        usage_snapshot: ApiUsageSnapshot | None = None,
        usage_error: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("高级设置")
        self.setModal(True)
        self.resize(540, 620)

        self._api_base_url = api_base_url
        self._api_key = api_key

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

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

        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems(["Auto", "1K", "2K", "4K"])
        self.image_size_combo.setCurrentText(generation_settings.image_size)

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

        form.addRow("模型", self.model_edit)
        form.addRow("Temperature", self.temperature_spin)
        form.addRow("Top P", self.top_p_spin)
        form.addRow("比例", self.aspect_ratio_combo)
        form.addRow("分辨率", self.image_size_combo)
        form.addRow("超时", self.timeout_spin)
        form.addRow("每组生成数量", self.variants_spin)
        form.addRow(self.seed_enabled_check, self.base_seed_spin)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addLayout(form)
        layout.addWidget(self._build_usage_details_group(usage_snapshot, usage_error))
        layout.addStretch(1)
        layout.addWidget(button_box)

    def _build_usage_details_group(
        self,
        usage_snapshot: ApiUsageSnapshot | None,
        usage_error: str,
    ) -> QGroupBox:
        group = QGroupBox("额度详情")
        detail_layout = QFormLayout(group)
        detail_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        detail_layout.setSpacing(10)

        self.usage_name_value = QLabel()
        self.usage_available_value = QLabel()
        self.usage_granted_value = QLabel()
        self.usage_used_value = QLabel()
        self.usage_remaining_value = QLabel()
        self.usage_unlimited_value = QLabel()
        self.usage_expire_value = QLabel()
        self.usage_limits_enabled_value = QLabel()
        self.usage_error_value = QLabel()
        self.usage_error_value.setWordWrap(True)
        self.usage_error_value.setObjectName("mutedLabel")

        self.usage_model_limits_edit = QPlainTextEdit()
        self.usage_model_limits_edit.setReadOnly(True)
        self.usage_model_limits_edit.setFixedHeight(100)
        self.usage_model_limits_edit.setPlaceholderText("未提供模型限制")

        detail_layout.addRow("Token 名称", self.usage_name_value)
        detail_layout.addRow("总可用", self.usage_available_value)
        detail_layout.addRow("总授予", self.usage_granted_value)
        detail_layout.addRow("已使用", self.usage_used_value)
        detail_layout.addRow("剩余", self.usage_remaining_value)
        detail_layout.addRow("无限额度", self.usage_unlimited_value)
        detail_layout.addRow("过期时间", self.usage_expire_value)
        detail_layout.addRow("模型限制", self.usage_limits_enabled_value)
        detail_layout.addRow("模型明细", self.usage_model_limits_edit)
        detail_layout.addRow("状态", self.usage_error_value)

        self._render_usage_details(usage_snapshot, usage_error)
        return group

    def _render_usage_details(
        self,
        usage_snapshot: ApiUsageSnapshot | None,
        usage_error: str,
    ) -> None:
        if usage_snapshot is None:
            fallback_text = "请先在主界面刷新额度"
            self.usage_name_value.setText("未查询")
            self.usage_available_value.setText("--")
            self.usage_granted_value.setText("--")
            self.usage_used_value.setText("--")
            self.usage_remaining_value.setText("--")
            self.usage_unlimited_value.setText("否")
            self.usage_expire_value.setText("不过期或未提供")
            self.usage_limits_enabled_value.setText("未启用")
            self.usage_model_limits_edit.setPlainText("未提供模型限制")
            self.usage_error_value.setText(usage_error or fallback_text)
            return

        self.usage_name_value.setText(usage_snapshot.name or "未命名")
        self.usage_available_value.setText(
            format_quota_value(usage_snapshot.total_available)
        )
        self.usage_granted_value.setText(
            format_quota_value(usage_snapshot.total_granted)
        )
        self.usage_used_value.setText(format_quota_value(usage_snapshot.total_used))
        self.usage_remaining_value.setText(
            format_quota_value(usage_snapshot.remaining_quota)
        )
        self.usage_unlimited_value.setText("是" if usage_snapshot.unlimited_quota else "否")
        self.usage_expire_value.setText(format_expires_at(usage_snapshot.expires_at))
        self.usage_limits_enabled_value.setText(
            "已启用" if usage_snapshot.model_limits_enabled else "未启用"
        )
        self.usage_model_limits_edit.setPlainText(
            format_model_limits(usage_snapshot.model_limits)
        )
        self.usage_error_value.setText(usage_error or "最近刷新成功")

    def get_values(self) -> tuple[str, str, GenerationSettings]:
        return (
            self._api_base_url,
            self._api_key,
        GenerationSettings(
            prompt="",
            model_type=self.model_edit.text().strip(),
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            aspect_ratio=self.aspect_ratio_combo.currentText(),
            image_size=self.image_size_combo.currentText(),
            timeout=self.timeout_spin.value(),
            variants_per_group=self.variants_spin.value(),
            seed_enabled=self.seed_enabled_check.isChecked(),
            base_seed=self.base_seed_spin.value(),
        ),
        )


class GenerationLogDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("图片生成历史日志")
        self.resize(1220, 780)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())

        self.entries: list[GenerationLogEntry] = []
        self.current_entry: GenerationLogEntry | None = None
        self._current_preview_path = ""

        layout = QVBoxLayout(self)
        description_label = QLabel("查看历次图片生成任务的摘要、明细和结果预览。")
        description_label.setObjectName("mutedLabel")
        layout.addWidget(description_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_title = QLabel("历史任务")
        left_title.setObjectName("logPanelTitle")
        self.history_count_label = QLabel("读取中...")
        self.history_count_label.setObjectName("mutedLabel")
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.currentRowChanged.connect(self._handle_entry_selected)
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.history_count_label)
        left_layout.addWidget(self.history_list, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.content_stack = QStackedWidget()

        detail_page = QWidget()
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(12)

        summary_group = QGroupBox("任务摘要")
        summary_group.setObjectName("logSummaryGroup")
        summary_layout = QGridLayout(summary_group)
        summary_layout.setHorizontalSpacing(18)
        summary_layout.setVerticalSpacing(8)

        self.summary_status_badge = QLabel("未加载")
        self.summary_status_badge.setObjectName("logStatusBadge")
        self.summary_started_value = QLabel("--")
        self.summary_finished_value = QLabel("--")
        self.summary_counts_value = QLabel("--")
        self.summary_output_dir_value = QLabel("--")
        self.summary_model_value = QLabel("--")
        self.summary_params_value = QLabel("--")

        summary_layout.addWidget(QLabel("状态"), 0, 0)
        summary_layout.addWidget(self.summary_status_badge, 0, 1)
        summary_layout.addWidget(QLabel("开始时间"), 0, 2)
        summary_layout.addWidget(self.summary_started_value, 0, 3)
        summary_layout.addWidget(QLabel("结束时间"), 1, 0)
        summary_layout.addWidget(self.summary_finished_value, 1, 1)
        summary_layout.addWidget(QLabel("执行统计"), 1, 2)
        summary_layout.addWidget(self.summary_counts_value, 1, 3)
        summary_layout.addWidget(QLabel("输出目录"), 2, 0)
        summary_layout.addWidget(self.summary_output_dir_value, 2, 1, 1, 3)
        summary_layout.addWidget(QLabel("模型"), 3, 0)
        summary_layout.addWidget(self.summary_model_value, 3, 1)
        summary_layout.addWidget(QLabel("参数"), 3, 2)
        summary_layout.addWidget(self.summary_params_value, 3, 3)

        detail_group = QGroupBox("执行明细")
        detail_group.setObjectName("logDetailGroup")
        detail_group_layout = QVBoxLayout(detail_group)
        detail_group_layout.setSpacing(10)

        self.item_table = QTableWidget(0, 6)
        self.item_table.setObjectName("historyItemTable")
        self.item_table.setHorizontalHeaderLabels(
            ["组序号", "匹配键", "变体", "状态", "耗时", "输出数"]
        )
        self.item_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.item_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.item_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.item_table.setAlternatingRowColors(True)
        self.item_table.verticalHeader().setVisible(False)
        header = self.item_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.item_table.currentCellChanged.connect(self._handle_item_changed)

        self.item_detail_edit = QPlainTextEdit()
        self.item_detail_edit.setReadOnly(True)
        self.item_detail_edit.setPlaceholderText("选择明细后查看 prompt、错误信息和文件路径。")
        self.item_detail_edit.setFixedHeight(170)

        detail_group_layout.addWidget(self.item_table)
        detail_group_layout.addWidget(self.item_detail_edit)

        preview_group = QGroupBox("结果预览")
        preview_group.setObjectName("logPreviewGroup")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_hint_label = QLabel("选择成功记录后可预览生成结果。")
        self.preview_hint_label.setObjectName("mutedLabel")
        self.log_preview_label = QLabel("暂无预览")
        self.log_preview_label.setObjectName("historyPreviewLabel")
        self.log_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.log_preview_label.setMinimumHeight(260)
        self.log_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_layout.addWidget(self.preview_hint_label)
        preview_layout.addWidget(self.log_preview_label, 1)

        detail_layout.addWidget(summary_group)
        detail_layout.addWidget(detail_group, 2)
        detail_layout.addWidget(preview_group, 2)

        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.addStretch(1)
        empty_title = QLabel("暂无生成历史")
        empty_title.setObjectName("logEmptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_text = QLabel("开始一次批量生成后，历史任务会自动记录到这里。")
        empty_text.setObjectName("mutedLabel")
        empty_text.setWordWrap(True)
        empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_text)
        empty_layout.addStretch(1)

        self.content_stack.addWidget(detail_page)
        self.content_stack.addWidget(empty_page)
        right_layout.addWidget(self.content_stack, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.load_history()

    def load_history(self) -> None:
        self.history_list.clear()
        self.entries = load_generation_log_entries()
        self.history_count_label.setText(f"共 {len(self.entries)} 条历史任务")

        if not self.entries:
            self.current_entry = None
            self.content_stack.setCurrentIndex(1)
            return

        self.content_stack.setCurrentIndex(0)
        for entry in self.entries:
            item = QListWidgetItem(self._build_history_item_text(entry))
            item.setToolTip(entry.output_dir or "未设置输出目录")
            self.history_list.addItem(item)
        self.history_list.setCurrentRow(0)

    def _build_history_item_text(self, entry: GenerationLogEntry) -> str:
        return (
            f"{format_log_datetime(entry.started_at)}\n"
            f"{format_generation_status(entry.status)} | 成功 {entry.success_count} / "
            f"失败 {entry.failure_count} / 完成 {entry.completed}\n"
            f"{summarize_output_dir(entry.output_dir)}"
        )

    def _handle_entry_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.entries):
            return

        entry = self.entries[row]
        self.current_entry = entry
        self.content_stack.setCurrentIndex(0)
        self._render_summary(entry)
        self._render_items(entry)

    def _render_summary(self, entry: GenerationLogEntry) -> None:
        self.summary_status_badge.setText(format_generation_status(entry.status))
        self.summary_status_badge.setProperty("logStatus", entry.status)
        self.summary_status_badge.style().unpolish(self.summary_status_badge)
        self.summary_status_badge.style().polish(self.summary_status_badge)
        self.summary_started_value.setText(format_log_datetime(entry.started_at))
        self.summary_finished_value.setText(format_log_datetime(entry.finished_at))
        self.summary_counts_value.setText(
            f"计划 {entry.total_planned} / 完成 {entry.completed} / "
            f"成功 {entry.success_count} / 失败 {entry.failure_count}"
        )
        self.summary_output_dir_value.setText(entry.output_dir or "--")
        self.summary_output_dir_value.setToolTip(entry.output_dir or "")
        self.summary_model_value.setText(entry.model_type or "--")
        self.summary_params_value.setText(
            f"T={entry.temperature:.2f}  Top P={entry.top_p:.2f}  "
            f"比例={entry.aspect_ratio}  分辨率={entry.image_size}"
        )

    def _render_items(self, entry: GenerationLogEntry) -> None:
        self.item_table.setRowCount(len(entry.items))
        for row, item in enumerate(entry.items):
            self.item_table.setItem(row, 0, QTableWidgetItem(str(item.group_index + 1)))
            self.item_table.setItem(row, 1, QTableWidgetItem(item.match_key or "--"))
            self.item_table.setItem(
                row, 2, QTableWidgetItem(str(item.variant_index + 1))
            )
            self.item_table.setItem(
                row,
                3,
                QTableWidgetItem("成功" if item.success else "失败"),
            )
            self.item_table.setItem(
                row, 4, QTableWidgetItem(f"{item.elapsed_seconds:.1f}s")
            )
            self.item_table.setItem(
                row, 5, QTableWidgetItem(str(len(item.saved_paths)))
            )

        if entry.items:
            preferred_row = next(
                (index for index, item in enumerate(entry.items) if item.success),
                0,
            )
            self.item_table.setCurrentCell(preferred_row, 0)
        else:
            self.item_detail_edit.setPlainText("当前任务没有生成任何明细记录。")
            self._set_preview_message("当前任务没有可预览的结果。")

    def _handle_item_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if self.current_entry is None:
            return
        if current_row < 0 or current_row >= len(self.current_entry.items):
            return

        item = self.current_entry.items[current_row]
        detail_lines = [
            f"匹配键：{item.match_key or '--'}",
            f"组序号：{item.group_index + 1}",
            f"变体：{item.variant_index + 1}",
            f"状态：{'成功' if item.success else '失败'}",
            f"耗时：{item.elapsed_seconds:.1f}s",
            f"请求耗时：{item.request_seconds:.1f}s",
            f"Seed：{item.seed if item.seed is not None else '--'}",
            "",
            "Prompt:",
            item.prompt_text or "--",
            "",
            "参考图:",
            "\n".join(item.reference_images) if item.reference_images else "--",
            "",
            "输出图:",
            "\n".join(item.saved_paths) if item.saved_paths else "--",
        ]
        if item.error:
            detail_lines.extend(["", "错误信息:", item.error])
        self.item_detail_edit.setPlainText("\n".join(detail_lines))
        self._render_preview(item)

    def _render_preview(self, item: GenerationLogItem) -> None:
        if not item.success or not item.saved_paths:
            self._set_preview_message("该条记录未生成成功，暂无可预览图片。")
            return

        image_path = item.saved_paths[-1]
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._set_preview_message(f"图片文件不存在或无法加载：{image_path}")
            return

        scaled = pixmap.scaled(
            self.log_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._current_preview_path = image_path
        self.preview_hint_label.setText(image_path)
        self.log_preview_label.setText("")
        self.log_preview_label.setPixmap(scaled)

    def _set_preview_message(self, message: str) -> None:
        self._current_preview_path = ""
        self.preview_hint_label.setText("选择成功记录后可预览生成结果。")
        self.log_preview_label.setPixmap(QPixmap())
        self.log_preview_label.setText(message)


class MainWindow(QMainWindow):
    def __init__(self, license_manager: LicenseManager | None = None) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1000, 800)

        self.license_manager = license_manager
        self.thread: QThread | None = None
        self.worker: BatchWorker | None = None
        self.usage_thread: QThread | None = None
        self.usage_worker: ApiUsageWorker | None = None
        self.slot_widgets: list[FolderSlotWidget] = []
        self.generation_settings = GenerationSettings()
        self.usage_snapshot: ApiUsageSnapshot | None = None
        self.usage_error = ""
        self.usage_loading = False
        self._usage_request_auto = False

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
        subtitle_label = QLabel("按文件名匹配参考槽位，批量组图并调用 API 生成结果。")
        subtitle_label.setObjectName("mutedLabel")

        root_layout.addWidget(title_label)
        root_layout.addWidget(subtitle_label)

        three_panel_layout = QHBoxLayout()
        three_panel_layout.setSpacing(14)
        three_panel_layout.addWidget(self._build_slots_group(), 3)
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
        self.task_summary_label = QLabel("已启用 0 个槽位，完全匹配后可执行 0 组")
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
        self.api_url_edit.setText("https://wuaiapi.com")
        self.api_url_edit.setEchoMode(QLineEdit.EchoMode.Password)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("输入 API Key")

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("输入统一的提示词")
        self.prompt_edit.setFixedHeight(110)

        self.prompt_mode_combo = QComboBox()
        self.prompt_mode_combo.addItem("固定提示词", "fixed")
        self.prompt_mode_combo.addItem("提示词文本", "file")

        self.prompt_file_edit = QLineEdit()
        self.prompt_file_edit.setPlaceholderText("选择提示词文本，例如 test.txt")
        prompt_file_row, self.prompt_file_button = create_line_button_row(
            self.prompt_file_edit,
            "选择文件",
        )

        prompt_file_hint = QLabel("格式：文件名=提示词；支持空行和 # 注释。")
        prompt_file_hint.setObjectName("mutedLabel")
        prompt_file_hint.setWordWrap(True)

        prompt_file_page = QWidget()
        prompt_file_layout = QVBoxLayout(prompt_file_page)
        prompt_file_layout.setContentsMargins(0, 0, 0, 0)
        prompt_file_layout.setSpacing(8)
        prompt_file_layout.addWidget(prompt_file_row)
        prompt_file_layout.addWidget(prompt_file_hint)

        self.prompt_input_stack = QStackedWidget()
        self.prompt_input_stack.addWidget(self.prompt_edit)
        self.prompt_input_stack.addWidget(prompt_file_page)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录")
        output_row, self.output_dir_button = create_line_button_row(
            self.output_dir_edit,
            "选择目录",
        )

        basic_form.addRow("URL", self.api_url_edit)
        basic_form.addRow("API Key", self.api_key_edit)
        basic_form.addRow("Prompt 模式", self.prompt_mode_combo)
        basic_form.addRow("Prompt", self.prompt_input_stack)
        basic_form.addRow("输出目录", output_row)

        self.output_dir_button.clicked.connect(self.choose_output_directory)
        self.prompt_file_button.clicked.connect(self.choose_prompt_file)
        self.prompt_mode_combo.currentIndexChanged.connect(self.update_prompt_mode_ui)
        self.prompt_edit.textChanged.connect(self.update_task_summary)
        self.prompt_file_edit.textChanged.connect(self.update_task_summary)
        self.api_url_edit.editingFinished.connect(self.handle_usage_input_finished)
        self.api_key_edit.editingFinished.connect(self.handle_usage_input_finished)
        self.update_prompt_mode_ui()
        return group

    def _build_advanced_settings_group(self) -> QGroupBox:
        group = QGroupBox("高级设置")
        settings_layout = QVBoxLayout(group)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        self.usage_card = QFrame()
        self.usage_card.setObjectName("usageCard")
        usage_layout = QVBoxLayout(self.usage_card)
        usage_layout.setContentsMargins(14, 14, 14, 14)
        usage_layout.setSpacing(8)

        usage_header = QHBoxLayout()
        usage_header.setSpacing(8)
        usage_title = QLabel("API Key 额度")
        usage_title.setObjectName("usageTitleLabel")
        self.usage_state_badge = QLabel("未配置")
        self.usage_state_badge.setObjectName("usageStateBadge")
        usage_header.addWidget(usage_title)
        usage_header.addStretch(1)
        usage_header.addWidget(self.usage_state_badge)

        self.usage_value_label = QLabel("--")
        self.usage_value_label.setObjectName("usageValueLabel")
        self.usage_meta_label = QLabel("请先填写 URL 和 API Key")
        self.usage_meta_label.setWordWrap(True)
        self.usage_meta_label.setObjectName("usageMetaLabel")

        self.usage_progress_bar = QProgressBar()
        self.usage_progress_bar.setTextVisible(False)
        self.usage_progress_bar.setFixedHeight(10)
        self.usage_progress_bar.hide()

        self.usage_time_label = QLabel("最近刷新：未刷新")
        self.usage_time_label.setObjectName("usageMetaLabel")
        self.usage_error_label = QLabel("")
        self.usage_error_label.setObjectName("usageErrorLabel")
        self.usage_error_label.setWordWrap(True)
        self.usage_error_label.hide()

        self.usage_refresh_button = QPushButton("刷新额度")
        self.usage_refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)

        usage_layout.addLayout(usage_header)
        usage_layout.addWidget(self.usage_value_label)
        usage_layout.addWidget(self.usage_meta_label)
        usage_layout.addWidget(self.usage_progress_bar)
        usage_layout.addWidget(self.usage_time_label)
        usage_layout.addWidget(self.usage_error_label)
        usage_layout.addWidget(
            self.usage_refresh_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.settings_button = QPushButton("设置")
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)

        settings_layout.addWidget(self.usage_card)
        settings_layout.addStretch(1)
        settings_layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.usage_refresh_button.clicked.connect(self.refresh_usage)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("任务控制")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        self.start_button = QPushButton("开始批量生成")
        self.stop_button = QPushButton("安全停止")
        self.save_button = QPushButton("保存配置")
        self.view_logs_button = QPushButton("查看生成日志")
        for button in (
            self.start_button,
            self.stop_button,
            self.save_button,
            self.view_logs_button,
        ):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.view_logs_button)
        layout.addWidget(self.progress_bar, 1)

        self.start_button.clicked.connect(self.start_batch)
        self.stop_button.clicked.connect(self.stop_batch)
        self.save_button.clicked.connect(self.save_config)
        self.view_logs_button.clicked.connect(self.open_generation_logs)
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
            #usageCard {
                background: #fffcf6;
                border: 1px solid #e7dfcf;
                border-radius: 14px;
            }
            #usageTitleLabel {
                font-size: 14px;
                font-weight: 700;
            }
            #usageValueLabel {
                font-size: 26px;
                font-weight: 700;
                color: #1f2937;
            }
            #usageMetaLabel {
                color: #6b6253;
            }
            #usageErrorLabel {
                color: #b45309;
                font-weight: 600;
            }
            #usageStateBadge {
                border-radius: 9px;
                padding: 3px 10px;
                font-size: 12px;
                font-weight: 700;
                color: #ffffff;
            }
            #usageStateBadge[usageState="inactive"] {
                background: #a8a29e;
            }
            #usageStateBadge[usageState="loading"] {
                background: #2563eb;
            }
            #usageStateBadge[usageState="success"] {
                background: #15803d;
            }
            #usageStateBadge[usageState="warning"] {
                background: #b45309;
            }
            #usageStateBadge[usageState="error"] {
                background: #b91c1c;
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
            #historyPreviewLabel {
                background: #f3f8f7;
                border: 1px dashed #9fb9b3;
                border-radius: 14px;
                color: #49635d;
                padding: 16px;
            }
            #logPanelTitle {
                font-size: 15px;
                font-weight: 700;
                color: #134e4a;
            }
            #logEmptyTitle {
                font-size: 22px;
                font-weight: 700;
                color: #134e4a;
            }
            #logStatusBadge {
                border-radius: 10px;
                padding: 4px 12px;
                font-weight: 700;
                color: #ffffff;
                background: #78716c;
            }
            #logStatusBadge[logStatus="success"] {
                background: #0f766e;
            }
            #logStatusBadge[logStatus="partial"] {
                background: #b45309;
            }
            #logStatusBadge[logStatus="failed"] {
                background: #b91c1c;
            }
            #logStatusBadge[logStatus="stopped"] {
                background: #a16207;
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
            QListWidget, QTableWidget {
                background: #ffffff;
                border: 1px solid #d5d0c3;
                border-radius: 10px;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #ede7da;
            }
            QListWidget::item:selected {
                background: #e6f5f2;
                color: #134e4a;
                border-left: 3px solid #0d9488;
            }
            QTableWidget {
                gridline-color: #ece5d8;
                selection-background-color: #e6f5f2;
                selection-color: #134e4a;
            }
            QHeaderView::section {
                background: #f6f1e7;
                color: #4b5563;
                border: none;
                border-bottom: 1px solid #e4ded0;
                padding: 8px 10px;
                font-weight: 600;
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

    def choose_prompt_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择提示词文本",
            self.prompt_file_edit.text().strip() or str(Path.home()),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if file_path:
            self.prompt_file_edit.setText(file_path)

    def current_prompt_mode(self) -> str:
        prompt_mode = self.prompt_mode_combo.currentData()
        return prompt_mode if prompt_mode in {"fixed", "file"} else "fixed"

    def set_prompt_mode(self, prompt_mode: str) -> None:
        target_mode = prompt_mode if prompt_mode in {"fixed", "file"} else "fixed"
        index = self.prompt_mode_combo.findData(target_mode)
        if index < 0:
            index = 0
        self.prompt_mode_combo.setCurrentIndex(index)
        self.update_prompt_mode_ui()

    def update_prompt_mode_ui(self) -> None:
        self.prompt_input_stack.setCurrentIndex(
            1 if self.current_prompt_mode() == "file" else 0
        )
        self.update_task_summary()

    def load_config(self) -> None:
        config = ConfigManager.load()

        self.api_url_edit.setText(config.api_base_url)
        self.api_key_edit.setText(config.api_key)
        self.output_dir_edit.setText(config.output_dir)

        self.generation_settings = config.generation_settings
        self.prompt_edit.setPlainText(self.generation_settings.prompt)
        self.prompt_file_edit.setText(self.generation_settings.prompt_file_path)
        self.set_prompt_mode(self.generation_settings.prompt_mode)

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
        self.refresh_usage(auto=True)

    def collect_config(self) -> AppConfig:
        settings = GenerationSettings(
            prompt=self.prompt_edit.toPlainText().strip(),
            prompt_mode=self.current_prompt_mode(),
            prompt_file_path=self.prompt_file_edit.text().strip(),
            model_type=self.generation_settings.model_type,
            temperature=self.generation_settings.temperature,
            top_p=self.generation_settings.top_p,
            aspect_ratio=self.generation_settings.aspect_ratio,
            image_size=self.generation_settings.image_size,
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

    def handle_usage_input_finished(self) -> None:
        self.refresh_usage(auto=True)

    def refresh_usage(self, auto: bool = False) -> None:
        base_url = self.api_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()

        if self.usage_thread is not None:
            if not auto:
                self.statusBar().showMessage("额度查询中，请稍候。", 3000)
            return

        self._usage_request_auto = auto
        if not base_url or not api_key:
            self.usage_snapshot = None
            self.usage_loading = False
            self.usage_error = "请先填写 URL 和 API Key。"
            self.render_usage_state()
            return

        self.usage_loading = True
        self.usage_error = ""
        self.render_usage_state()

        self.usage_thread = QThread(self)
        self.usage_worker = ApiUsageWorker(base_url, api_key, timeout=10)
        self.usage_worker.moveToThread(self.usage_thread)

        self.usage_thread.started.connect(self.usage_worker.run)
        self.usage_worker.usage_loaded.connect(self.apply_usage_snapshot)
        self.usage_worker.usage_failed.connect(self.apply_usage_error)
        self.usage_worker.finished.connect(self.usage_thread.quit)
        self.usage_worker.finished.connect(self.usage_worker.deleteLater)
        self.usage_thread.finished.connect(self.usage_thread.deleteLater)
        self.usage_thread.finished.connect(self.handle_usage_worker_finished)
        self.usage_thread.start()

    def apply_usage_snapshot(self, snapshot: object) -> None:
        self.usage_snapshot = snapshot if isinstance(snapshot, ApiUsageSnapshot) else None
        self.usage_loading = False
        self.usage_error = ""
        self.render_usage_state()
        self.statusBar().showMessage("额度信息已刷新。", 3000)

    def apply_usage_error(self, message: str) -> None:
        self.usage_loading = False
        self.usage_error = message
        self.render_usage_state()
        if not self._usage_request_auto:
            self.statusBar().showMessage(message, 4000)

    def handle_usage_worker_finished(self) -> None:
        self.usage_thread = None
        self.usage_worker = None

    def set_usage_badge(self, text: str, state: str) -> None:
        self.usage_state_badge.setText(text)
        self.usage_state_badge.setProperty("usageState", state)
        self.usage_state_badge.style().unpolish(self.usage_state_badge)
        self.usage_state_badge.style().polish(self.usage_state_badge)
        self.usage_state_badge.update()

    def render_usage_state(self) -> None:
        snapshot = self.usage_snapshot
        self.usage_refresh_button.setEnabled(not self.usage_loading)
        self.usage_error_label.setVisible(bool(self.usage_error))
        self.usage_error_label.setText(self.usage_error)

        if self.usage_loading:
            self.set_usage_badge("查询中", "loading")
            self.usage_value_label.setText("额度查询中…")
            self.usage_meta_label.setText("正在获取最新额度信息，请稍候。")
            if snapshot is None:
                self.usage_time_label.setText("最近刷新：未刷新")
                self.usage_progress_bar.hide()
            return

        if snapshot is None:
            if self.api_url_edit.text().strip() and self.api_key_edit.text().strip():
                self.set_usage_badge("查询失败", "error")
                self.usage_value_label.setText("--")
                self.usage_meta_label.setText("暂未获取到额度数据。")
            else:
                self.set_usage_badge("未配置", "inactive")
                self.usage_value_label.setText("--")
                self.usage_meta_label.setText("请先填写 URL 和 API Key")
            self.usage_time_label.setText("最近刷新：未刷新")
            self.usage_progress_bar.hide()
            return

        if snapshot.unlimited_quota:
            self.usage_value_label.setText("无限额度")
            self.usage_meta_label.setText(
                f"已使用 {format_quota_value(snapshot.total_used)}"
            )
            self.usage_progress_bar.hide()
        else:
            self.usage_value_label.setText(
                format_quota_value(snapshot.remaining_quota)
            )
            self.usage_meta_label.setText(
                "已使用 {used} / 总授予 {granted}".format(
                    used=format_quota_value(snapshot.total_used),
                    granted=format_quota_value(snapshot.total_granted),
                )
            )
            if snapshot.total_granted > 0:
                progress_ratio = min(
                    max(snapshot.total_used / snapshot.total_granted, 0.0),
                    1.0,
                )
                self.usage_progress_bar.setRange(0, 1000)
                self.usage_progress_bar.setValue(int(progress_ratio * 1000))
                self.usage_progress_bar.show()
            else:
                self.usage_progress_bar.hide()

        if self.usage_error:
            self.set_usage_badge("刷新失败", "warning")
        else:
            self.set_usage_badge("已同步", "success")

        self.usage_time_label.setText(
            f"最近刷新：{format_fetched_at(snapshot.fetched_at)}"
        )

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(
            api_base_url=self.api_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            generation_settings=self.generation_settings,
            usage_snapshot=self.usage_snapshot,
            usage_error=self.usage_error,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        api_base_url, api_key, generation_settings = dialog.get_values()
        generation_settings.prompt = self.prompt_edit.toPlainText().strip()
        generation_settings.prompt_mode = self.current_prompt_mode()
        generation_settings.prompt_file_path = self.prompt_file_edit.text().strip()
        self.api_url_edit.setText(api_base_url)
        self.api_key_edit.setText(api_key)
        self.generation_settings = generation_settings
        self.update_task_summary()
        self.refresh_usage(auto=True)

    def open_generation_logs(self) -> None:
        dialog = GenerationLogDialog(self)
        dialog.exec()

    def create_generation_log_entry(self, summary: dict[str, object]) -> GenerationLogEntry:
        items_data = summary.get("items", [])
        log_items: list[GenerationLogItem] = []
        if isinstance(items_data, list):
            for item_data in items_data:
                if not isinstance(item_data, dict):
                    continue
                try:
                    log_items.append(GenerationLogItem.from_dict(item_data))
                except (TypeError, ValueError):
                    continue

        return GenerationLogEntry(
            id=str(uuid4()),
            started_at=str(summary.get("started_at", "")),
            finished_at=str(summary.get("finished_at", "")),
            status=str(summary.get("status", "failed")),
            total_planned=int(summary.get("total", 0)),
            completed=int(summary.get("completed", 0)),
            success_count=int(summary.get("success_count", 0)),
            failure_count=int(summary.get("failure_count", 0)),
            stopped=bool(summary.get("stopped", False)),
            output_dir=str(summary.get("output_dir", "")),
            api_base_url=str(summary.get("api_base_url", "")),
            model_type=str(summary.get("model_type", "")),
            temperature=float(summary.get("temperature", 0.0)),
            top_p=float(summary.get("top_p", 0.0)),
            aspect_ratio=str(summary.get("aspect_ratio", "")),
            image_size=str(summary.get("image_size", "")),
            variants_per_group=int(summary.get("variants_per_group", 1)),
            items=log_items,
        )

    def update_task_summary(self) -> None:
        enabled_slot_sources: list[tuple[str, dict[str, str]]] = []
        enabled_slots = 0
        invalid_slot_found = False
        for widget in self.slot_widgets:
            slot = widget.to_slot()
            widget.refresh_count()
            if not slot.enabled:
                continue

            enabled_slots += 1
            if not slot.path:
                invalid_slot_found = True
                continue

            folder_path = Path(slot.path)
            if not folder_path.exists() or not folder_path.is_dir():
                invalid_slot_found = True
                continue

            image_files = list_image_files(folder_path)
            if not image_files:
                invalid_slot_found = True
                continue

            image_map, duplicate_errors = build_image_name_map(image_files)
            if duplicate_errors or not image_map:
                invalid_slot_found = True
                continue

            enabled_slot_sources.append((slot.name, image_map))

        if enabled_slots == 0:
            self.task_summary_label.setText("已启用 0 个槽位，完全匹配后可执行 0 组")
            return

        if invalid_slot_found or len(enabled_slot_sources) != enabled_slots:
            summary_text = f"已启用 {enabled_slots} 个槽位，可执行 0 组"
        else:
            all_keys = set().union(*(image_map.keys() for _, image_map in enabled_slot_sources))
            aligned = all(set(image_map) == all_keys for _, image_map in enabled_slot_sources)
            if not aligned:
                summary_text = f"已启用 {enabled_slots} 个槽位，文件名未对齐"
            else:
                summary_text = (
                    f"已启用 {enabled_slots} 个槽位，完全匹配后可执行 {len(all_keys)} 组"
                )

            if self.current_prompt_mode() == "file":
                prompt_path = self.prompt_file_edit.text().strip()
                prompt_valid = False
                if prompt_path:
                    prompt_file = Path(prompt_path)
                    if prompt_file.exists() and prompt_file.is_file():
                        prompt_map, format_errors = parse_prompt_text_file(prompt_file)
                        prompt_valid = not format_errors and set(prompt_map) == all_keys
                if not prompt_valid:
                    summary_text = f"{summary_text}，提示词文本待修正"

        self.task_summary_label.setText(summary_text)

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
        base_errors: list[str] = []
        slot_errors: list[str] = []
        prompt_file_errors: list[str] = []
        format_errors: list[str] = []

        if not config_data.api_base_url:
            base_errors.append("请填写 API URL。")
        if not config_data.api_key:
            base_errors.append("请填写 API Key。")
        if not config_data.output_dir:
            base_errors.append("请选择输出目录。")

        prompt_map: dict[str, str] | None = None
        if settings.prompt_mode == "file":
            prompt_file_path = settings.prompt_file_path.strip()
            if not prompt_file_path:
                prompt_file_errors.append("请选择提示词文本文件。")
            else:
                prompt_path = Path(prompt_file_path)
                if not prompt_path.exists():
                    prompt_file_errors.append(
                        f"提示词文本不存在：{settings.prompt_file_path}"
                    )
                elif not prompt_path.is_file():
                    prompt_file_errors.append(
                        f"提示词文本不是有效文件：{settings.prompt_file_path}"
                    )
                else:
                    prompt_map, parse_errors = parse_prompt_text_file(prompt_path)
                    format_errors.extend(
                        [f"提示词文件第 {error[2:]}" if error.startswith("第 ") else error for error in parse_errors]
                    )
        elif not settings.prompt:
            base_errors.append("请填写固定 Prompt。")

        enabled_slot_count = 0
        enabled_slot_sources: list[tuple[str, dict[str, str]]] = []
        for slot in config_data.folder_slots:
            if not slot.enabled:
                continue

            enabled_slot_count += 1
            if not slot.path:
                slot_errors.append(f"槽位“{slot.name}”未选择文件夹。")
                continue

            folder_path = Path(slot.path)
            if not folder_path.exists() or not folder_path.is_dir():
                slot_errors.append(f"槽位“{slot.name}”的目录不存在。")
                continue

            image_files = list_image_files(folder_path)
            if not image_files:
                slot_errors.append(f"槽位“{slot.name}”目录中没有可用图片。")
                continue

            image_map, duplicate_errors = build_image_name_map(image_files)
            if duplicate_errors:
                slot_errors.extend(
                    [f"槽位“{slot.name}”：{error}" for error in duplicate_errors]
                )
                continue

            enabled_slot_sources.append((slot.name, image_map))

        if enabled_slot_count == 0:
            slot_errors.append("至少启用一个有效的参考槽位。")

        expected_keys: list[str] = []
        if enabled_slot_sources:
            all_keys = set().union(
                *(set(image_map.keys()) for _, image_map in enabled_slot_sources)
            )
            common_keys = set(all_keys)
            for _, image_map in enabled_slot_sources:
                common_keys &= set(image_map.keys())

            expected_keys = sort_match_keys(list(all_keys))
            for slot_name, image_map in enabled_slot_sources:
                slot_keys = set(image_map.keys())
                missing_keys = sort_match_keys(list(all_keys - slot_keys))
                extra_keys = sort_match_keys(list(slot_keys - common_keys))
                if missing_keys:
                    slot_errors.append(
                        f"槽位“{slot_name}”缺少文件名：{', '.join(missing_keys)}"
                    )
                if extra_keys:
                    slot_errors.append(
                        f"槽位“{slot_name}”多出文件名：{', '.join(extra_keys)}"
                    )

        if prompt_map is not None and expected_keys:
            prompt_keys = set(prompt_map.keys())
            expected_key_set = set(expected_keys)
            missing_prompts = sort_match_keys(list(expected_key_set - prompt_keys))
            extra_prompts = sort_match_keys(list(prompt_keys - expected_key_set))
            if missing_prompts:
                prompt_file_errors.append(
                    f"提示词文本缺少文件名：{', '.join(missing_prompts)}"
                )
            if extra_prompts:
                prompt_file_errors.append(
                    f"提示词文本多出文件名：{', '.join(extra_prompts)}"
                )

        sections: list[str] = []
        if base_errors:
            sections.append("基础配置问题：\n" + "\n".join(f"- {error}" for error in base_errors))
        if slot_errors:
            sections.append("槽位目录问题：\n" + "\n".join(f"- {error}" for error in slot_errors))
        if prompt_file_errors:
            sections.append(
                "提示词文件问题：\n" + "\n".join(f"- {error}" for error in prompt_file_errors)
            )
        if format_errors:
            sections.append("格式问题：\n" + "\n".join(f"- {error}" for error in format_errors))
        if sections:
            raise ValueError("\n\n".join(sections))

        tasks = build_batch_tasks(
            enabled_slot_sources,
            fixed_prompt=settings.prompt,
            prompt_map=prompt_map,
        )
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
            self.prompt_mode_combo,
            self.prompt_edit,
            self.prompt_file_edit,
            self.prompt_file_button,
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

        try:
            log_entry = self.create_generation_log_entry(data)
            append_generation_log_entry(log_entry)
        except Exception as exc:
            self.append_log(f"历史日志保存失败：{exc}")

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
    app = QApplication(sys.argv)
    startup_exit_code, license_manager = authorize_before_launch()
    if startup_exit_code is not None:
        return startup_exit_code

    window = MainWindow(license_manager=license_manager)
    window.show()
    return app.exec()


def authorize_before_launch() -> tuple[int | None, LicenseManager | None]:
    return run_license_authorization(
        dialog_factory=LicenseLoginDialog,
        show_error=lambda message: QMessageBox.critical(None, "启动失败", message),
        accepted_code=int(QDialog.DialogCode.Accepted),
    )


if __name__ == "__main__":
    raise SystemExit(main())
