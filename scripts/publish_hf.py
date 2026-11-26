#!/usr/bin/env python3
"""Export, validate, and publish anonymized build_process_samples to Hugging Face.

All GCP coordinates and the anonymization salt are read from environment variables
so no private infrastructure details are stored in git or printed to logs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SQL_PATH = REPO_ROOT / "sql" / "export_samples.sql"
CARD_TEMPLATE_PATH = REPO_ROOT / "hf" / "README.md.tmpl"

ALLOWED_PROCESS_NAMES = frozenset(
    {"GradleDaemon", "KotlinCompileDaemon", "GradleWorkerMain"}
)
HASHED_RUN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
GCS_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
VERSION_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")

EXPECTED_COLUMNS: list[tuple[str, str]] = [
    ("run_id", "string"),
    ("sample_timestamp", "timestamp"),
    ("elapsed_time", "int64"),
    ("pid", "string"),
    ("name", "string"),
    ("heap_used", "int64"),
    ("heap_cap", "int64"),
    ("rss", "int64"),
    ("gc_time", "int64"),
    ("jit_compiled_methods", "int64"),
    ("jit_failed_compilations", "int64"),
    ("jit_invalidated_compilations", "int64"),
    ("jit_compilation_time_ms", "int64"),
    ("classes_loaded", "int64"),
    ("classes_unloaded", "int64"),
    ("class_load_time_ms", "int64"),
    ("run_finished_at", "timestamp"),
]


@dataclass
class DatasetStats:
    total_rows: int
    distinct_runs: int
    min_date: str
    max_date: str
    total_size_mb: float
    shard_count: int
    jit_coverage_pct: float
    class_coverage_pct: float
    process_rows: dict[str, int]
    process_runs: dict[str, int]


def render_export_sql(sql_template: str, bq_table: str, gcs_uri: str) -> str:
    if not BQ_IDENTIFIER_RE.match(bq_table):
        raise ValueError("Invalid BigQuery table identifier format.")
    if not gcs_uri.startswith("gs://") or not gcs_uri.endswith("*.parquet"):
        raise ValueError("Invalid GCS export URI format.")
    return (
        sql_template.replace("__BQ_TABLE__", bq_table).replace("__GCS_URI__", gcs_uri)
    )


def _check_arrow_type(field: pa.Field, expected: str) -> None:
    if expected == "string" and not (
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ):
        raise ValueError(f"Column {field.name} must be string, got {field.type}")
    elif expected == "int64" and not pa.types.is_int64(field.type):
        raise ValueError(f"Column {field.name} must be int64, got {field.type}")
    elif expected == "timestamp" and not pa.types.is_timestamp(field.type):
        raise ValueError(f"Column {field.name} must be timestamp, got {field.type}")


def validate_parquet_files(
    parquet_files: Iterable[Path], *, min_runs: int = 1
) -> DatasetStats:
    files = sorted(Path(p) for p in parquet_files)
    if not files:
        raise ValueError("No Parquet files found to validate.")

    total_rows = 0
    total_bytes = 0
    all_runs: set[str] = set()
    process_rows: dict[str, int] = {name: 0 for name in sorted(ALLOWED_PROCESS_NAMES)}
    process_runs: dict[str, set[str]] = {
        name: set() for name in sorted(ALLOWED_PROCESS_NAMES)
    }
    jit_non_null = 0
    class_non_null = 0
    min_ts = None
    max_ts = None
    seen_keys: set[tuple[str, str, int]] = set()

    expected_names = [col for col, _ in EXPECTED_COLUMNS]

    for path in files:
        total_bytes += path.stat().st_size
        table = pq.read_table(path)

        if table.column_names != expected_names:
            raise ValueError(
                f"Schema column mismatch in {path.name}: got {table.column_names}"
            )
        for col_name, expected_type in EXPECTED_COLUMNS:
            _check_arrow_type(table.schema.field(col_name), expected_type)

        if table.num_rows == 0:
            continue

        total_rows += table.num_rows

        # Required columns must not contain nulls
        for req_col in (
            "run_id",
            "sample_timestamp",
            "elapsed_time",
            "pid",
            "name",
            "heap_used",
            "heap_cap",
            "rss",
            "gc_time",
            "run_finished_at",
        ):
            if table[req_col].null_count > 0:
                raise ValueError(f"Column {req_col} contains nulls in {path.name}")

        # Metric bounds checks
        for non_neg in ("heap_used", "heap_cap", "gc_time", "elapsed_time"):
            if pc.any(pc.less(table[non_neg], 0)).as_py():
                raise ValueError(f"Column {non_neg} contains negative values in {path.name}")
        if pc.any(pc.less_equal(table["rss"], 0)).as_py():
            raise ValueError(f"Column rss contains <= 0 values in {path.name}")

        # Verify anonymized run_id format and allowed process names
        unique_run_ids = pc.unique(table["run_id"]).to_pylist()
        for rid in unique_run_ids:
            if not isinstance(rid, str) or not HASHED_RUN_ID_RE.match(rid):
                raise ValueError(
                    "Anonymization check failed: run_id is not a 16-char lowercase hex hash."
                )
            all_runs.add(rid)

        unique_names = set(pc.unique(table["name"]).to_pylist())
        unexpected_names = unique_names - ALLOWED_PROCESS_NAMES
        if unexpected_names:
            raise ValueError(f"Unexpected process names found: {sorted(unexpected_names)}")

        # Coverage & timestamp bounds
        jit_non_null += table.num_rows - table["jit_compiled_methods"].null_count
        class_non_null += table.num_rows - table["classes_loaded"].null_count

        file_min = pc.min(table["sample_timestamp"]).as_py()
        file_max = pc.max(table["sample_timestamp"]).as_py()
        if min_ts is None or file_min < min_ts:
            min_ts = file_min
        if max_ts is None or file_max > max_ts:
            max_ts = file_max

        # Duplicate check & per-process breakdown
        rids = table["run_id"].to_pylist()
        pids = table["pid"].to_pylist()
        names = table["name"].to_pylist()
        ts_us = pc.cast(table["sample_timestamp"], pa.int64()).to_pylist()
        for rid, pid, proc_name, ts_val in zip(rids, pids, names, ts_us):
            key = (rid, pid, ts_val)
            if key in seen_keys:
                raise ValueError(f"Duplicate (run_id, pid, sample_timestamp) detected in {path.name}")
            seen_keys.add(key)
            process_rows[proc_name] += 1
            process_runs[proc_name].add(rid)

    if total_rows == 0:
        raise ValueError("Dataset validation failed: 0 rows exported.")
    if len(all_runs) < min_runs:
        raise ValueError(
            f"Dataset validation failed: found {len(all_runs)} runs, expected >= {min_runs}."
        )

    return DatasetStats(
        total_rows=total_rows,
        distinct_runs=len(all_runs),
        min_date=min_ts.strftime("%Y-%m-%d"),
        max_date=max_ts.strftime("%Y-%m-%d"),
        total_size_mb=round(total_bytes / (1024 * 1024), 2),
        shard_count=len(files),
        jit_coverage_pct=round((jit_non_null / total_rows) * 100, 1),
        class_coverage_pct=round((class_non_null / total_rows) * 100, 1),
        process_rows=process_rows,
        process_runs={k: len(v) for k, v in process_runs.items()},
    )


def render_dataset_card(template_text: str, stats: DatasetStats, version_tag: str) -> str:
    replacements = {
        "{{VERSION_TAG}}": version_tag,
        "{{TOTAL_ROWS}}": f"{stats.total_rows:,}",
        "{{DISTINCT_RUNS}}": f"{stats.distinct_runs:,}",
        "{{MIN_DATE}}": stats.min_date,
        "{{MAX_DATE}}": stats.max_date,
        "{{TOTAL_SIZE_MB}}": f"{stats.total_size_mb:.1f}",
        "{{SHARD_COUNT}}": str(stats.shard_count),
        "{{JIT_COVERAGE_PCT}}": f"{stats.jit_coverage_pct:.1f}",
        "{{CLASS_COVERAGE_PCT}}": f"{stats.class_coverage_pct:.1f}",
        "{{GRADLE_DAEMON_ROWS}}": f"{stats.process_rows.get('GradleDaemon', 0):,}",
        "{{GRADLE_DAEMON_RUNS}}": f"{stats.process_runs.get('GradleDaemon', 0):,}",
        "{{KOTLIN_DAEMON_ROWS}}": f"{stats.process_rows.get('KotlinCompileDaemon', 0):,}",
        "{{KOTLIN_DAEMON_RUNS}}": f"{stats.process_runs.get('KotlinCompileDaemon', 0):,}",
        "{{GRADLE_WORKER_ROWS}}": f"{stats.process_rows.get('GradleWorkerMain', 0):,}",
        "{{GRADLE_WORKER_RUNS}}": f"{stats.process_runs.get('GradleWorkerMain', 0):,}",
    }
    rendered = template_text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def export_from_bigquery(
    *,
    project_id: str,
    dataset: str,
    table: str,
    bucket_name: str,
    salt: str,
    min_samples_per_run: int,
    local_data_dir: Path,
) -> list[Path]:
    from google.cloud import bigquery, storage

    if not GCS_BUCKET_RE.match(bucket_name):
        raise ValueError("Invalid GCS_EXPORT_BUCKET name.")

    export_id = uuid.uuid4().hex[:12]
    gcs_prefix = f"hf-exports/{export_id}/"
    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}train-*.parquet"
    bq_table = f"{project_id}.{dataset}.{table}"

    sql_template = SQL_PATH.read_text(encoding="utf-8")
    query_sql = render_export_sql(sql_template, bq_table, gcs_uri)

    bq_client = bigquery.Client(project=project_id)
    gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.bucket(bucket_name)

    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id_salt", "STRING", salt),
                bigquery.ScalarQueryParameter(
                    "min_samples_per_run", "INT64", min_samples_per_run
                ),
            ]
        )
        print("Running BigQuery EXPORT DATA job...")
        job = bq_client.query(query_sql, job_config=job_config)
        job.result()

        blobs = sorted(
            (b for b in bucket.list_blobs(prefix=gcs_prefix) if b.name.endswith(".parquet")),
            key=lambda b: b.name,
        )
        if not blobs:
            raise RuntimeError("BigQuery EXPORT DATA produced no Parquet files.")

        local_data_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        total = len(blobs)
        for idx, blob in enumerate(blobs):
            target = local_data_dir / f"train-{idx:05d}-of-{total:05d}.parquet"
            blob.download_to_filename(str(target))
            downloaded.append(target)
        print(f"Downloaded {len(downloaded)} Parquet shard(s) for validation.")
        return downloaded
    finally:
        # Clean up temporary staging blobs in GCS even if download/validation fails
        try:
            staging_blobs = list(bucket.list_blobs(prefix=gcs_prefix))
            if staging_blobs:
                bucket.delete_blobs(staging_blobs)
                print(f"Cleaned up {len(staging_blobs)} temporary staging blob(s).")
        except Exception as cleanup_err:
            print(f"Warning: failed to clean up temporary staging blobs: {cleanup_err}", file=sys.stderr)


def upload_to_huggingface(
    *,
    stage_dir: Path,
    repo_id: str,
    token: str,
    version_tag: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    commit_message = f"Publish dataset snapshot {version_tag}"
    print(f"Uploading dataset folder to Hugging Face ({repo_id})...")
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        delete_patterns=["data/*.parquet"],
    )
    if version_tag:
        print(f"Creating dataset tag {version_tag}...")
        api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=version_tag,
            tag_message=commit_message,
            exist_ok=True,
        )
    print(f"Published {repo_id} ({version_tag}).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export, validate, and publish build_process_samples to Hugging Face."
    )
    parser.add_argument(
        "--hf-repo-id",
        default=os.environ.get("HF_REPO_ID", "inakivillar/build-process-dataset"),
        help="Target Hugging Face dataset repository ID (default: inakivillar/build-process-dataset)",
    )
    parser.add_argument(
        "--version-tag",
        default=os.environ.get("VERSION_TAG", "v1.0.0"),
        help="Dataset release tag to stamp in README and HF git tags (e.g. v1.0.0)",
    )
    parser.add_argument(
        "--min-samples-per-run",
        type=int,
        default=3,
        help="Minimum deduplicated sample rows required to keep a run (default: 3)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=10,
        help="Validation gate: minimum distinct runs required before upload (default: 10)",
    )
    parser.add_argument(
        "--local-parquet-dir",
        type=Path,
        default=None,
        help="Skip BigQuery export and validate/upload Parquet files from this local directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export and validate without uploading to Hugging Face",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not VERSION_TAG_RE.match(args.version_tag):
        raise ValueError(f"Invalid --version-tag '{args.version_tag}'; expected semver like v1.0.0")

    with tempfile.TemporaryDirectory(prefix="bpw-hf-stage-") as tmp:
        stage_dir = Path(tmp)
        data_dir = stage_dir / "data"

        if args.local_parquet_dir:
            src_files = sorted(args.local_parquet_dir.glob("*.parquet"))
            if not src_files:
                raise ValueError(f"No .parquet files found in {args.local_parquet_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
            parquet_files: list[Path] = []
            for idx, src in enumerate(src_files):
                dst = data_dir / f"train-{idx:05d}-of-{len(src_files):05d}.parquet"
                shutil.copy2(src, dst)
                parquet_files.append(dst)
        else:
            project_id = os.environ.get("BQ_PROJECT_ID", "").strip()
            dataset = os.environ.get("BQ_DATASET", "").strip()
            table = os.environ.get("BQ_TABLE", "build_process_samples").strip()
            bucket = os.environ.get("GCS_EXPORT_BUCKET", "").strip()
            salt = os.environ.get("RUN_ID_HASH_SALT", "")

            missing = [
                name
                for name, val in [
                    ("BQ_PROJECT_ID", project_id),
                    ("BQ_DATASET", dataset),
                    ("GCS_EXPORT_BUCKET", bucket),
                    ("RUN_ID_HASH_SALT", salt),
                ]
                if not val
            ]
            if missing:
                raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

            parquet_files = export_from_bigquery(
                project_id=project_id,
                dataset=dataset,
                table=table,
                bucket_name=bucket,
                salt=salt,
                min_samples_per_run=args.min_samples_per_run,
                local_data_dir=data_dir,
            )

        stats = validate_parquet_files(parquet_files, min_runs=args.min_runs)
        card_text = render_dataset_card(
            CARD_TEMPLATE_PATH.read_text(encoding="utf-8"),
            stats=stats,
            version_tag=args.version_tag,
        )
        (stage_dir / "README.md").write_text(card_text, encoding="utf-8")

        summary_lines = [
            "Validation passed:",
            f"  Version:      {args.version_tag}",
            f"  Rows:         {stats.total_rows:,}",
            f"  Runs:         {stats.distinct_runs:,}",
            f"  Date range:   {stats.min_date} to {stats.max_date}",
            f"  Parquet size: {stats.total_size_mb:.1f} MB ({stats.shard_count} shard(s))",
            f"  JIT / Class:  {stats.jit_coverage_pct:.1f}% / {stats.class_coverage_pct:.1f}%",
        ]
        summary = "\n".join(summary_lines)
        print(summary)

        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a", encoding="utf-8") as fh:
                fh.write(f"### Dataset Export Summary ({'dry-run' if args.dry_run else args.version_tag})\n\n")
                fh.write(f"```\n{summary}\n```\n")

        if args.dry_run:
            print("Dry run enabled; skipping Hugging Face upload.")
            return 0

        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            raise RuntimeError("Missing required environment variable: HF_TOKEN")

        upload_to_huggingface(
            stage_dir=stage_dir,
            repo_id=args.hf_repo_id,
            token=hf_token,
            version_tag=args.version_tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
