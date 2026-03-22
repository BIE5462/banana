from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generation_log_service import (
    GenerationLogEntry,
    GenerationLogItem,
    append_entry,
    load_entries,
)


class GenerationLogServiceTests(unittest.TestCase):
    def build_entry(self, entry_id: str, *, status: str = "success") -> GenerationLogEntry:
        return GenerationLogEntry(
            id=entry_id,
            started_at="2026-03-22T10:00:00+08:00",
            finished_at="2026-03-22T10:01:00+08:00",
            status=status,
            total_planned=2,
            completed=2,
            success_count=2 if status == "success" else 1,
            failure_count=0 if status == "success" else 1,
            stopped=False,
            output_dir="D:/output",
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
                    match_key="demo",
                    prompt_text="make it shine",
                    variant_index=0,
                    success=True,
                    reference_images=["D:/refs/a.png"],
                    saved_paths=["D:/output/result.png"],
                    elapsed_seconds=1.2,
                    request_seconds=0.9,
                    seed=1,
                )
            ],
        )

    def test_load_entries_returns_empty_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "generation_logs.json"
            self.assertEqual(load_entries(path), [])

    def test_load_entries_returns_empty_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "generation_logs.json"
            path.write_text("{not-json}", encoding="utf-8")

            self.assertEqual(load_entries(path), [])

    def test_load_entries_skips_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "generation_logs.json"
            valid_entry = self.build_entry("entry-1").to_dict()
            path.write_text(
                json.dumps([valid_entry, {"broken": True}, "bad-record"], ensure_ascii=False),
                encoding="utf-8",
            )

            entries = load_entries(path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "entry-1")

    def test_append_entry_prepends_latest_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "generation_logs.json"
            append_entry(self.build_entry("first"), path)
            append_entry(self.build_entry("second", status="partial"), path)

            entries = load_entries(path)

        self.assertEqual([entry.id for entry in entries], ["second", "first"])
        self.assertEqual(entries[0].status, "partial")


if __name__ == "__main__":
    unittest.main()
