# build-process-dataset

Reproducible BigQuery-to-Parquet export, validation, and Hugging Face Hub publishing pipeline for the [`cdsap/build-process-dataset`](https://huggingface.co/datasets/cdsap/build-process-dataset) JVM build telemetry dataset (collected with [Build Process Watcher](https://github.com/cdsap/build-process-watcher)).

## What It Does

1. **Exports & cleans** `build_process_samples` directly from BigQuery to Snappy Parquet in GCS (`sql/export_samples.sql`):
   - Keeps standard CI runs (`^run-\d+$`) with at least 3 samples across `GradleDaemon`, `KotlinCompileDaemon`, and `GradleWorkerMain`.
   - Deduplicates on `(run_id, pid, sample_timestamp)`.
   - Filters out negative heap/cap/gc/elapsed glitches and `rss <= 0` process-exit rows.
   - Replaces `run_id` with a deterministic 16-char salted SHA-256 hex digest (`@run_id_salt` query parameter).
2. **Validates** every Parquet shard locally (`scripts/publish_hf.py`):
   - Exact 17-column Arrow schema and nullable `int64` types for optional `jstat` columns.
   - Confirms every `run_id` matches `^[0-9a-f]{16}$` (zero raw IDs).
   - Confirms zero duplicate keys, zero negative metrics, and only expected process names.
3. **Publishes** to Hugging Face (`cdsap/build-process-dataset`, `CC0-1.0`) with an auto-updated Dataset Card (`hf/README.md.tmpl`) and semver git tag (`v1.0.0`, `v1.1.0`, …), then cleans up temporary GCS staging files.

## One-Time Setup

### 1. GCS Staging Bucket (US region)
Create a temporary staging bucket in the same region as your BigQuery dataset (`US`), with a 1-day lifecycle rule as a safety net:

```bash
gcloud storage buckets create gs://<YOUR_BUCKET> --project=<YOUR_PROJECT> --location=US
```

### 2. Service Account Permissions
Grant a least-privilege GCP service account:
- `roles/bigquery.jobUser` on `<YOUR_PROJECT>`
- `roles/bigquery.dataViewer` on `<YOUR_DATASET>`
- `roles/storage.objectAdmin` on `gs://<YOUR_BUCKET>`

### 3. Hugging Face Dataset Repo & Token
1. Create [`cdsap/build-process-dataset`](https://huggingface.co/new-dataset) on Hugging Face (start **Private** to verify the Dataset Viewer, then flip **Public**).
2. Create a fine-grained Hugging Face write token scoped only to `cdsap/build-process-dataset`.

### 4. GitHub Repository Secrets
Add these under **Settings → Secrets and variables → Actions**:

| Secret | Description |
| --- | --- |
| `GCP_WIF_PROVIDER` & `GCP_SERVICE_ACCOUNT` | Workload Identity Federation provider & SA email (or set `GCP_SA_KEY` JSON instead) |
| `BQ_PROJECT_ID` | GCP project containing the BigQuery dataset |
| `BQ_DATASET` | BigQuery dataset ID |
| `GCS_EXPORT_BUCKET` | Staging GCS bucket name (without `gs://`) |
| `RUN_ID_HASH_SALT` | Fixed secret salt (`openssl rand -hex 32`); keep stable across versions |
| `HF_TOKEN` | Fine-grained Hugging Face write token |

## Running a Release

### Via GitHub Actions (Recommended)
1. Go to **Actions → Publish Hugging Face Dataset → Run workflow**.
2. Run once with `dry_run: true` and `version_tag: v1.0.0` to verify row/run counts and validation checks in the job summary.
3. Run again with `dry_run: false` and `version_tag: v1.0.0` to publish and tag on Hugging Face.

### Locally (Using Your Existing `gcloud` Login)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export BQ_PROJECT_ID="..."
export BQ_DATASET="..."
export GCS_EXPORT_BUCKET="..."
export RUN_ID_HASH_SALT="..."
export HF_TOKEN="..."

# Dry run first
python scripts/publish_hf.py --version-tag v1.0.0 --dry-run

# Publish
python scripts/publish_hf.py --version-tag v1.0.0
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

- Repository code: MIT (`LICENSE`)
- Published dataset: CC0 1.0 Universal (`cc0-1.0`)
