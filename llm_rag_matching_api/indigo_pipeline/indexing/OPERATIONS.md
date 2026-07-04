# Indigo Indexing Operations

## Storage Model

- ChromaDB is the final store.
- Store operations are append-only. Existing IDs are skipped and are not cleared by normal pipeline runs.
- Use a new `INDIGO_RAG_STORE_DIR` version, such as `/app/data/rag_store_v5`, for a clean rebuild.

## Checkpoints And Artifacts

- `extraction_{doc_type}_checkpoint.json` is a temporary crash-recovery checkpoint.
- `index_{doc_type}.pkl` is a legacy full-run checkpoint and is not used by split runner retries.
- Split run artifacts under `data/checkpoints/split_index_runs/*/{doc_type}/artifacts/*.json` are reusable extraction cache files.
- The builder does not automatically read legacy extraction checkpoints for reuse.
- The split runner removes `extraction_{doc_type}_checkpoint.json` after a successful doc-type run unless `--keep-extraction-checkpoints` is passed.
- `--retry-failed` reads failed doc IDs from logs but reloads documents from the current train data instead of trusting `index_{doc_type}.pkl`.

## Resume Rules

- Use `--resume-extract` only with the same `--run-dir`, unchanged train data, and unchanged extraction code.
- Do not use resume after collection, filtering, doc-id logic, prompt, or extraction code changes.
- After data/code changes, start a new run directory.
- When `--resume-extract` is used, the runner reuses an artifact only if the artifact docs match the current batch doc IDs and text fingerprints.
- If a same-numbered batch artifact exists but the current batch contents changed, the runner re-extracts that batch.

## Artifact Reuse Rules

- Normal incremental runs may keep split artifacts.
- Full re-extraction for one doc type should remove that doc type's old split artifacts first:

```bash
sudo rm -rf data/checkpoints/split_index_runs/*/project
```

- Full clean rebuilds should use a new rag store directory and remove old artifacts for the doc types being rebuilt.

## Retention

Use one of these policies to avoid unbounded split artifact growth:

```bash
python3 -m indigo_pipeline.indexing.split_runner --retention-days 14
python3 -m indigo_pipeline.indexing.split_runner --max-runs 5
```

Or configure defaults in `.env`:

```bash
INDIGO_SPLIT_RUN_RETENTION_DAYS=14
INDIGO_SPLIT_RUN_MAX_RUNS=5
```

`--cleanup-success` removes per-batch docs/artifacts for the current successful doc type. Use it only when you do not need artifact reuse for that run.
