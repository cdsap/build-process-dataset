-- Curated, anonymized export of build_process_samples to Parquet in GCS.
-- Placeholders __BQ_TABLE__ and __GCS_URI__ are validated and substituted by
-- scripts/publish_hf.py. The salt is passed strictly as the @run_id_salt query
-- parameter and never interpolated into SQL text.

EXPORT DATA
OPTIONS (
  uri = '__GCS_URI__',
  format = 'PARQUET',
  compression = 'SNAPPY',
  overwrite = true
) AS
WITH cleaned AS (
  SELECT
    run_id,
    sample_timestamp,
    elapsed_time,
    pid,
    name,
    heap_used,
    heap_cap,
    rss,
    gc_time,
    jit_compiled_methods,
    jit_failed_compilations,
    jit_invalidated_compilations,
    jit_compilation_time_ms,
    classes_loaded,
    classes_unloaded,
    class_load_time_ms,
    run_finished_at
  FROM `__BQ_TABLE__`
  WHERE
    REGEXP_CONTAINS(run_id, r'^run-\d+$')
    AND name IN ('GradleDaemon', 'KotlinCompileDaemon', 'GradleWorkerMain')
    AND heap_used >= 0
    AND heap_cap >= 0
    AND rss > 0
    AND gc_time >= 0
    AND elapsed_time >= 0
    AND jit_compiled_methods IS NOT NULL
    AND jit_failed_compilations IS NOT NULL
    AND jit_invalidated_compilations IS NOT NULL
    AND jit_compilation_time_ms IS NOT NULL
    AND classes_loaded IS NOT NULL
    AND classes_unloaded IS NOT NULL
    AND class_load_time_ms IS NOT NULL
    AND jit_compiled_methods >= 0
    AND jit_failed_compilations >= 0
    AND jit_invalidated_compilations >= 0
    AND jit_compilation_time_ms >= 0
    AND classes_loaded >= 0
    AND classes_unloaded >= 0
    AND class_load_time_ms >= 0
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, pid, sample_timestamp
    ORDER BY run_finished_at DESC
  ) = 1
),
valid_runs AS (
  SELECT run_id
  FROM cleaned
  GROUP BY run_id
  HAVING COUNT(*) >= @min_samples_per_run
)
SELECT
  SUBSTR(TO_HEX(SHA256(CONCAT(c.run_id, @run_id_salt))), 1, 16) AS run_id,
  c.sample_timestamp,
  c.elapsed_time,
  c.pid,
  c.name,
  c.heap_used,
  c.heap_cap,
  c.rss,
  c.gc_time,
  c.jit_compiled_methods,
  c.jit_failed_compilations,
  c.jit_invalidated_compilations,
  c.jit_compilation_time_ms,
  c.classes_loaded,
  c.classes_unloaded,
  c.class_load_time_ms,
  c.run_finished_at
FROM cleaned AS c
INNER JOIN valid_runs AS v
  ON c.run_id = v.run_id
ORDER BY
  c.run_finished_at,
  run_id,
  c.pid,
  c.elapsed_time;
