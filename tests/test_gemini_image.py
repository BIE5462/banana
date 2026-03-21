from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from GeminiImage import (
    GeminiImageGenerator,
    build_image_name_map,
    build_batch_tasks,
    list_image_files,
    parse_prompt_text_file,
    normalize_match_key,
    sanitize_filename_part,
    summarize_reference_images,
)


class GeminiImageTests(unittest.TestCase):
    def test_list_image_files_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.png").write_bytes(b"png")
            (root / "a.jpg").write_bytes(b"jpg")
            (root / "note.txt").write_text("skip", encoding="utf-8")

            image_files = list_image_files(root)
            self.assertEqual(
                image_files,
                [str(root / "a.jpg"), str(root / "b.png")],
            )

    def test_normalize_match_key_uses_stem(self) -> None:
        self.assertEqual(normalize_match_key("look-01.png"), "look-01")
        self.assertEqual(normalize_match_key(" look-02 "), "look-02")

    def test_build_image_name_map_detects_duplicate_stems(self) -> None:
        image_map, errors = build_image_name_map(["C:/tmp/a.png", "C:/tmp/a.jpg"])

        self.assertEqual(Path(image_map["a"]), Path("C:/tmp/a.png"))
        self.assertEqual(len(errors), 1)
        self.assertIn("主文件名“a”重复", errors[0])

    def test_build_batch_tasks_matches_by_stem_and_fixed_prompt(self) -> None:
        tasks = build_batch_tasks(
            [
                ("饰品", {"look-01": "bracelet/look-01.png", "look-02": "bracelet/look-02.png"}),
                ("模特", {"look-01": "model/look-01.jpg", "look-02": "model/look-02.jpg"}),
                ("手链", {"look-01": "chain/look-01.webp", "look-02": "chain/look-02.webp"}),
            ],
            fixed_prompt="统一提示词",
        )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].match_key, "look-01")
        self.assertEqual(tasks[0].prompt_text, "统一提示词")
        self.assertEqual(
            tasks[0].reference_images,
            ["bracelet/look-01.png", "model/look-01.jpg", "chain/look-01.webp"],
        )
        self.assertEqual(tasks[1].match_key, "look-02")
        self.assertEqual(tasks[1].prompt_text, "统一提示词")

    def test_build_batch_tasks_uses_prompt_map(self) -> None:
        tasks = build_batch_tasks(
            [
                ("饰品", {"look-01": "bracelet/look-01.png", "look-02": "bracelet/look-02.png"}),
                ("模特", {"look-01": "model/look-01.jpg", "look-02": "model/look-02.jpg"}),
            ],
            prompt_map={"look-01": "提示词一", "look-02": "提示词二"},
        )

        self.assertEqual([task.prompt_text for task in tasks], ["提示词一", "提示词二"])

    def test_parse_prompt_text_file_parses_comments_and_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text(
                "# 注释\n\nlook-01.png=金色手链\nlook-02 = 银色手链\n",
                encoding="utf-8",
            )

            prompt_map, errors = parse_prompt_text_file(prompt_path)

        self.assertEqual(errors, [])
        self.assertEqual(
            prompt_map,
            {"look-01": "金色手链", "look-02": "银色手链"},
        )

    def test_parse_prompt_text_file_reports_format_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text(
                "no-separator\n=missing-name\nlook-01=\nlook-01=first\nlook-01=second\n",
                encoding="utf-8",
            )

            _, errors = parse_prompt_text_file(prompt_path)

        self.assertEqual(len(errors), 4)
        self.assertIn("第 1 行缺少“=”分隔符。", errors)
        self.assertIn("第 2 行文件名为空。", errors)
        self.assertIn("第 3 行提示词为空。", errors)
        self.assertIn("第 5 行文件名“look-01”重复。", errors)

    def test_sanitize_and_summary_for_output_name(self) -> None:
        sanitized = sanitize_filename_part(" model / test * name ")
        summary = summarize_reference_images(
            ["C:/tmp/bracelet gold.png", "C:/tmp/model#01.jpg"]
        )

        self.assertEqual(sanitized, "model-test-name")
        self.assertIn("bracelet-gold", summary)
        self.assertIn("model-01", summary)

    def test_create_request_data_includes_temperature_top_p_and_seed(self) -> None:
        generator = GeminiImageGenerator("https://example.com", "demo-key")

        with patch.object(generator, "image_to_base64", return_value="encoded"):
            request_data = generator.create_request_data(
                prompt="test prompt",
                seed=7,
                aspect_ratio="16:9",
                image_size="4K",
                temperature=1.1,
                top_p=0.55,
                image_paths=["demo.png"],
            )

        generation_config = request_data["generationConfig"]
        self.assertEqual(generation_config["temperature"], 1.1)
        self.assertEqual(generation_config["topP"], 0.55)
        self.assertEqual(generation_config["seed"], 7)
        self.assertEqual(generation_config["imageConfig"]["aspectRatio"], "16:9")
        self.assertEqual(generation_config["imageConfig"]["imageSize"], "4K")
        self.assertEqual(len(request_data["contents"][0]["parts"]), 2)

    def test_create_request_data_uses_default_ratio_when_only_image_size_is_set(self) -> None:
        generator = GeminiImageGenerator("https://example.com", "demo-key")

        request_data = generator.create_request_data(
            prompt="test prompt",
            seed=None,
            aspect_ratio="Auto",
            image_size="1K",
        )

        image_config = request_data["generationConfig"]["imageConfig"]
        self.assertEqual(image_config["aspectRatio"], "1:1")
        self.assertEqual(image_config["imageSize"], "1K")


if __name__ == "__main__":
    unittest.main()
