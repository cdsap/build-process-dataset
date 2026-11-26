import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.publish_hf import (
    CARD_TEMPLATE_PATH,
    SQL_PATH,
    render_dataset_card,
    render_export_sql,
    validate_parquet_files,
)


def _sample_table(overrides: dict | None = None) -> pa.Table:
    now = dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    later = dt.datetime(2026, 10, 1, 12, 5, 0, tzinfo=dt.timezone.utc)
    data = {
        "run_id": ["0123456789abcdef", "0123456789abcdef", "fedcba9876543210"],
        "sample_timestamp": [now, now + dt.timedelta(seconds=5), later],
        "elapsed_time": [5, 10, 5],
        "pid": ["1001", "1001", "2002"],
        "name": ["GradleDaemon", "GradleDaemon", "KotlinCompileDaemon"],
        "heap_used": [420, 480, 310],
        "heap_cap": [1024, 1024, 512],
        "rss": [900, 950, 620],
        "gc_time": [120, 140, 80],
        "jit_compiled_methods": [1500, 1620, 900],
        "jit_failed_compilations": [0, 0, 0],
        "jit_invalidated_compilations": [0, 0, 0],
        "jit_compilation_time_ms": [3200, 3400, 1800],
        "classes_loaded": [12000, 12450, 8100],
        "classes_unloaded": [10, 10, 0],
        "class_load_time_ms": [2100, 2180, 1100],
        "run_finished_at": [later, later, later],
    }
    if overrides:
        data.update(overrides)
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("sample_timestamp", pa.timestamp("us", tz="UTC")),
            ("elapsed_time", pa.int64()),
            ("pid", pa.string()),
            ("name", pa.string()),
            ("heap_used", pa.int64()),
            ("heap_cap", pa.int64()),
            ("rss", pa.int64()),
            ("gc_time", pa.int64()),
            ("jit_compiled_methods", pa.int64()),
            ("jit_failed_compilations", pa.int64()),
            ("jit_invalidated_compilations", pa.int64()),
            ("jit_compilation_time_ms", pa.int64()),
            ("classes_loaded", pa.int64()),
            ("classes_unloaded", pa.int64()),
            ("class_load_time_ms", pa.int64()),
            ("run_finished_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    return pa.Table.from_pydict(data, schema=schema)


class PublishHfTests(unittest.TestCase):
    def test_valid_parquet_and_dataset_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "train-00000-of-00001.parquet"
            pq.write_table(_sample_table(), p)
            stats = validate_parquet_files([p], min_runs=2)
            self.assertEqual(stats.total_rows, 3)
            self.assertEqual(stats.distinct_runs, 2)
            self.assertEqual(stats.min_date, "2026-10-01")
            self.assertEqual(stats.max_date, "2026-10-01")
            self.assertAlmostEqual(stats.jit_coverage_pct, 100.0)
            self.assertAlmostEqual(stats.class_coverage_pct, 100.0)
            self.assertEqual(stats.process_rows["GradleDaemon"], 2)
            self.assertEqual(stats.process_rows["KotlinCompileDaemon"], 1)

            card = render_dataset_card(
                CARD_TEMPLATE_PATH.read_text(encoding="utf-8"),
                stats=stats,
                version_tag="v1.0.0",
            )
            self.assertIn("license: cc0-1.0", card)
            self.assertNotIn("{{", card)
            self.assertIn("`v1.0.0`", card)

    def test_rejects_unhashed_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.parquet"
            pq.write_table(
                _sample_table({"run_id": ["run-1783046073708", "0123456789abcdef", "fedcba9876543210"]}),
                p,
            )
            with self.assertRaisesRegex(ValueError, "Anonymization check failed"):
                validate_parquet_files([p], min_runs=1)

    def test_rejects_negative_heap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.parquet"
            pq.write_table(_sample_table({"heap_used": [-5, 480, 310]}), p)
            with self.assertRaisesRegex(ValueError, "negative values"):
                validate_parquet_files([p], min_runs=1)

    def test_rejects_zero_rss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.parquet"
            pq.write_table(_sample_table({"rss": [0, 950, 620]}), p)
            with self.assertRaisesRegex(ValueError, "rss contains <= 0"):
                validate_parquet_files([p], min_runs=1)

    def test_rejects_null_jit_or_class_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p_jit = Path(tmp) / "null_jit.parquet"
            pq.write_table(_sample_table({"jit_compiled_methods": [1500, None, 900]}), p_jit)
            with self.assertRaisesRegex(ValueError, "jit_compiled_methods contains nulls"):
                validate_parquet_files([p_jit], min_runs=1)

            p_cls = Path(tmp) / "null_cls.parquet"
            pq.write_table(_sample_table({"classes_loaded": [12000, 12450, None]}), p_cls)
            with self.assertRaisesRegex(ValueError, "classes_loaded contains nulls"):
                validate_parquet_files([p_cls], min_runs=1)

    def test_rejects_duplicate_sample_key(self) -> None:
        now = dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dup.parquet"
            pq.write_table(
                _sample_table({"sample_timestamp": [now, now, now + dt.timedelta(seconds=5)]}),
                p,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_parquet_files([p], min_runs=1)

    def test_render_export_sql_validates_identifiers(self) -> None:
        sql = SQL_PATH.read_text(encoding="utf-8")
        rendered = render_export_sql(
            sql,
            "my-project.my_dataset.build_process_samples",
            "gs://my-bucket/hf-exports/abc/train-*.parquet",
        )
        self.assertIn("`my-project.my_dataset.build_process_samples`", rendered)
        self.assertIn("@run_id_salt", rendered)
        with self.assertRaises(ValueError):
            render_export_sql(sql, "bad; DROP TABLE x", "gs://my-bucket/train-*.parquet")


if __name__ == "__main__":
    unittest.main()
