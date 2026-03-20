from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from GeminiImage import (
    GeminiImageGenerator,
    build_batch_tasks,
    list_image_files,
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

    def test_build_batch_tasks_uses_minimum_length(self) -> None:
        tasks = build_batch_tasks(
            [
                ("饰品", ["a1.png", "a2.png", "a3.png"]),
                ("模特", ["m1.png", "m2.png"]),
                ("手链", ["b1.png", "b2.png", "b3.png"]),
            ]
        )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].reference_images, ["a1.png", "m1.png", "b1.png"])
        self.assertEqual(tasks[1].reference_images, ["a2.png", "m2.png", "b2.png"])

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
                temperature=1.1,
                top_p=0.55,
                image_paths=["demo.png"],
            )

        generation_config = request_data["generationConfig"]
        self.assertEqual(generation_config["temperature"], 1.1)
        self.assertEqual(generation_config["topP"], 0.55)
        self.assertEqual(generation_config["seed"], 7)
        self.assertEqual(generation_config["imageConfig"]["aspectRatio"], "16:9")
        self.assertEqual(len(request_data["contents"][0]["parts"]), 2)


if __name__ == "__main__":
    unittest.main()
