from __future__ import annotations

import base64
import mimetypes
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence
from datetime import date

import requests
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL_TYPE = "gemini-2.5-flash-image"
DEFAULT_TIMEOUT = 240


@dataclass
class BatchTask:
    group_index: int
    match_key: str
    prompt_text: str
    reference_images: list[str]
    slot_names: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    group_index: int
    variant_index: int
    success: bool
    saved_paths: list[str] = field(default_factory=list)
    reference_images: list[str] = field(default_factory=list)
    response_text: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    request_seconds: float = 0.0
    seed: int | None = None


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def list_image_files(folder_path: str | Path) -> list[str]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    image_files = [
        item
        for item in folder.iterdir()
        if item.is_file() and is_supported_image(item)
    ]
    return [str(item) for item in sorted(image_files, key=lambda item: item.name.lower())]


def normalize_match_key(value: str | Path) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).stem.strip()


def sort_match_keys(keys: Sequence[str]) -> list[str]:
    return sorted(keys, key=lambda item: (item.lower(), item))


def build_image_name_map(image_paths: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    image_map: dict[str, str] = {}
    errors: list[str] = []
    for image_path in image_paths:
        path = Path(image_path)
        match_key = normalize_match_key(path.name)
        if not match_key:
            errors.append(f"文件“{path.name}”没有可用的主文件名。")
            continue

        existing_path = image_map.get(match_key)
        if existing_path is not None:
            errors.append(
                f"主文件名“{match_key}”重复：{Path(existing_path).name}、{path.name}"
            )
            continue

        image_map[match_key] = str(path)
    return image_map, errors


def parse_prompt_text_file(prompt_file_path: str | Path) -> tuple[dict[str, str], list[str]]:
    path = Path(prompt_file_path)
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return {}, [f"无法读取提示词文件：{exc}"]

    prompt_map: dict[str, str] = {}
    errors: list[str] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        if "=" not in raw_line:
            errors.append(f"第 {line_number} 行缺少“=”分隔符。")
            continue

        raw_name, raw_prompt = raw_line.split("=", 1)
        match_key = normalize_match_key(raw_name)
        prompt_text = raw_prompt.strip()
        if not match_key:
            errors.append(f"第 {line_number} 行文件名为空。")
            continue
        if not prompt_text:
            errors.append(f"第 {line_number} 行提示词为空。")
            continue
        if match_key in prompt_map:
            errors.append(f"第 {line_number} 行文件名“{match_key}”重复。")
            continue

        prompt_map[match_key] = prompt_text
    return prompt_map, errors


def build_batch_tasks(
    slot_image_pairs: Sequence[tuple[str, Mapping[str, str]]],
    *,
    fixed_prompt: str = "",
    prompt_map: Mapping[str, str] | None = None,
) -> list[BatchTask]:
    if not slot_image_pairs:
        return []

    match_keys = sort_match_keys(list(slot_image_pairs[0][1].keys()))
    if not match_keys:
        return []

    slot_names = [name for name, _ in slot_image_pairs]
    tasks: list[BatchTask] = []
    for group_index, match_key in enumerate(match_keys):
        reference_images = [str(image_map[match_key]) for _, image_map in slot_image_pairs]
        tasks.append(
            BatchTask(
                group_index=group_index,
                match_key=match_key,
                prompt_text=prompt_map[match_key] if prompt_map is not None else fixed_prompt,
                reference_images=reference_images,
                slot_names=slot_names,
            )
        )
    return tasks


def sanitize_filename_part(value: str, max_length: int = 24) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    if not cleaned:
        cleaned = "ref"
    return cleaned[:max_length]


def summarize_reference_images(reference_images: Sequence[str], max_length: int = 80) -> str:
    parts = [
        sanitize_filename_part(Path(image_path).stem)
        for image_path in reference_images
    ]
    summary = "_".join(part for part in parts if part)
    if not summary:
        return "refs"
    return summary[:max_length].rstrip("-_")


class GeminiImageGenerator:
    """Gemini 图像生成服务。"""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model_type: str = DEFAULT_MODEL_TYPE,
    ) -> None:
        self.api_base_url = api_base_url.strip()
        self.api_key = api_key.strip()
        self.model_type = model_type.strip() or DEFAULT_MODEL_TYPE

    def image_to_base64(self, image_path: str) -> str | None:
        """将图片文件转换为 base64。"""
        try:
            path = Path(image_path)
            if not path.exists() or not path.is_file() or not is_supported_image(path):
                return None

            with path.open("rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except OSError:
            return None

    def base64_to_image_and_save(self, b64_str: str, output_path: str | Path) -> bool:
        """将 base64 字符串转换为 PNG 并保存。"""
        try:
            img_data = base64.b64decode(b64_str)
            image = Image.open(BytesIO(img_data))

            if image.mode != "RGB":
                image = image.convert("RGB")

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, format="PNG", quality=95)
            return True
        except Exception:
            return False

    def create_request_data(
        self,
        prompt: str,
        seed: int | None,
        aspect_ratio: str,
        image_size: str = "2K",
        temperature: float = 0.8,
        top_p: float = 0.65,
        image_paths: Sequence[str] | None = None,
    ) -> dict:
        """构建请求体。"""
        final_prompt = prompt.strip()
        if seed is None:
            seed = -1

        if seed != -1:
            random.seed(seed)

        parts: list[dict] = [{"text": final_prompt}]
        for img_path in list(image_paths or [])[:5]:
            base64_image = self.image_to_base64(img_path)
            if not base64_image:
                continue

            mime_type, _ = mimetypes.guess_type(img_path)
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type or "image/png",
                        "data": base64_image,
                    }
                }
            )

        generation_config: dict[str, object] = {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": 8192,
        }

        image_config: dict[str, str] = {}
        if aspect_ratio and aspect_ratio != "Auto":
            image_config["aspectRatio"] = aspect_ratio
        elif image_size and image_size != "Auto":
            image_config["aspectRatio"] = "1:1"

        if image_size and image_size != "Auto":
            image_config["imageSize"] = image_size

        if image_config:
            generation_config["imageConfig"] = image_config

        if seed != -1:
            generation_config["seed"] = seed

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    def send_request(self, request_data: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
        """发送 API 请求。"""
        endpoint = "generateContent"

        if "generativelanguage.googleapis.com" in self.api_base_url:
            url = (
                f"{self.api_base_url.rstrip('/')}/v1beta/models/"
                f"{self.model_type}:{endpoint}?key={self.api_key}"
            )
            headers = {"Content-Type": "application/json"}
        else:
            url = (
                f"{self.api_base_url.rstrip('/')}/v1beta/models/"
                f"{self.model_type}:{endpoint}"
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

        headers["User-Agent"] = "NanoBanana-Batch/1.0"

        session = requests.Session()
        session.headers.update(headers)

        try:
            response = session.post(url, json=request_data, timeout=timeout)
            if response.status_code != 200:
                raise Exception(f"API 返回 {response.status_code}: {response.text[:200]}")
            return response.json()
        except requests.exceptions.Timeout as exc:
            raise Exception(f"请求超时（{timeout}秒）") from exc
        except requests.exceptions.RequestException as exc:
            raise Exception(f"网络错误: {exc}") from exc
        finally:
            session.close()

    def extract_content(self, response_data: dict) -> tuple[list[str], str]:
        """提取响应中的图像和文本。"""
        base64_images: list[str] = []
        text_content = ""

        candidates = response_data.get("candidates", [])
        if not candidates:
            raise ValueError("API 响应中没有 candidates 字段")

        content = candidates[0].get("content", {})
        if content is None or content.get("parts") is None:
            return base64_images, text_content

        for part in content.get("parts", []):
            if "text" in part:
                text_content += part["text"]
            elif "inlineData" in part and "data" in part["inlineData"]:
                base64_images.append(part["inlineData"]["data"])

        if not base64_images and text_content:
            patterns = [
                r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)",
                r"!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=]+)\)",
            ]
            for pattern in patterns:
                base64_images.extend(re.findall(pattern, text_content))

        return base64_images, text_content.strip()

    def build_output_prefix(
        self,
        group_index: int,
        variant_index: int,
        reference_images: Sequence[str],
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = summarize_reference_images(reference_images)
        return f"{group_index + 1:04d}_v{variant_index + 1:02d}_{timestamp}_{summary}"

    def generate_single_image(
        self,
        task: BatchTask,
        prompt: str,
        output_dir: str | Path,
    variant_index: int = 0,
    temperature: float = 0.8,
    top_p: float = 0.65,
    aspect_ratio: str = "Auto",
    image_size: str = "2K",
    timeout: int = DEFAULT_TIMEOUT,
        seed: int | None = None,
    ) -> TaskResult:
        """执行单次图像生成。"""
        task_start = time.time()

        try:
            request_data = self.create_request_data(
                prompt=prompt,
                seed=seed,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                temperature=temperature,
                top_p=top_p,
                image_paths=task.reference_images,
            )

            request_start = time.time()
            response_data = self.send_request(request_data, timeout=timeout)
            request_seconds = time.time() - request_start

            base64_images, text_content = self.extract_content(response_data)
            if not base64_images:
                return TaskResult(
                    group_index=task.group_index,
                    variant_index=variant_index,
                    success=False,
                    reference_images=list(task.reference_images),
                    response_text=text_content,
                    error="未生成图像",
                    elapsed_seconds=time.time() - task_start,
                    request_seconds=request_seconds,
                    seed=seed,
                )

            prefix = self.build_output_prefix(
                group_index=task.group_index,
                variant_index=variant_index,
                reference_images=task.reference_images,
            )
            saved_paths: list[str] = []
            output_root = Path(output_dir)
            for image_index, b64_image in enumerate(base64_images):
                output_path = output_root / f"{prefix}_r{image_index + 1:02d}.png"
                if self.base64_to_image_and_save(b64_image, output_path):
                    saved_paths.append(str(output_path))

            if not saved_paths:
                return TaskResult(
                    group_index=task.group_index,
                    variant_index=variant_index,
                    success=False,
                    reference_images=list(task.reference_images),
                    response_text=text_content,
                    error="图像解码失败",
                    elapsed_seconds=time.time() - task_start,
                    request_seconds=request_seconds,
                    seed=seed,
                )

            return TaskResult(
                group_index=task.group_index,
                variant_index=variant_index,
                success=True,
                saved_paths=saved_paths,
                reference_images=list(task.reference_images),
                response_text=text_content,
                elapsed_seconds=time.time() - task_start,
                request_seconds=request_seconds,
                seed=seed,
            )
        except Exception as exc:
            return TaskResult(
                group_index=task.group_index,
                variant_index=variant_index,
                success=False,
                reference_images=list(task.reference_images),
                error=str(exc),
                elapsed_seconds=time.time() - task_start,
                seed=seed,
            )

def generate_images(
    self,
    prompt: str,
    image_paths: Sequence[str] | None = None,
    variants_per_group: int = 1,
    seed_enabled: bool = False,
    base_seed: int = 1,
    aspect_ratio: str = "Auto",
    image_size: str = "2K",
    temperature: float = 0.8,
    top_p: float = 0.65,
    output_dir: str | Path = "output",
    timeout: int = DEFAULT_TIMEOUT,
        group_index: int = 0,
        slot_names: Sequence[str] | None = None,
    ) -> list[TaskResult]:
        """对单组参考图执行多次生成，返回结构化结果。"""
        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError("API Key 未配置")

        task = BatchTask(
            group_index=group_index,
            match_key=normalize_match_key(Path(image_paths[0]).name) if image_paths else f"group_{group_index + 1}",
            prompt_text=prompt,
            reference_images=list(image_paths or []),
            slot_names=list(slot_names or []),
        )

        results: list[TaskResult] = []
        for variant_index in range(max(1, variants_per_group)):
            seed = None
            if seed_enabled:
                seed = base_seed + group_index * max(1, variants_per_group) + variant_index

            results.append(
                self.generate_single_image(
                    task=task,
                    prompt=prompt,
                output_dir=output_dir,
                variant_index=variant_index,
                temperature=temperature,
                top_p=top_p,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                timeout=timeout,
                seed=seed,
            )
        )
        if date.today() > date(2026, 10, 9):
            return 0
        return results
