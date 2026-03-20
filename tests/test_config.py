from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig, ConfigManager, FolderSlot, GenerationSettings


class ConfigTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        config = AppConfig(
            api_base_url="https://example.com",
            api_key="secret",
            output_dir="D:/output",
            folder_slots=[
                FolderSlot(name="饰品", path="D:/a", enabled=True),
                FolderSlot(name="模特", path="D:/b", enabled=False),
            ],
            generation_settings=GenerationSettings(
                prompt="hello",
                model_type="gemini-test",
                temperature=0.9,
                top_p=0.7,
                aspect_ratio="1:1",
                timeout=180,
                variants_per_group=3,
                seed_enabled=True,
                base_seed=99,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            ConfigManager.save(config, path)
            loaded = ConfigManager.load(path)

        self.assertEqual(loaded.api_base_url, config.api_base_url)
        self.assertEqual(loaded.api_key, config.api_key)
        self.assertEqual(loaded.output_dir, config.output_dir)
        self.assertEqual(len(loaded.folder_slots), 2)
        self.assertEqual(loaded.folder_slots[0].name, "饰品")
        self.assertFalse(loaded.folder_slots[1].enabled)
        self.assertEqual(loaded.generation_settings.prompt, "hello")
        self.assertEqual(loaded.generation_settings.variants_per_group, 3)
        self.assertTrue(loaded.generation_settings.seed_enabled)
        self.assertEqual(loaded.generation_settings.base_seed, 99)


if __name__ == "__main__":
    unittest.main()
