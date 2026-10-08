# Legacy attack outputs

Superseded layouts, kept for reference. Nothing here is read by the pipeline.

- `nested_results/` — the original doubly-nested results tree
  (`<dataset>/<model>/<recipe>/namesarnav_<ds>__test/<model>/<recipe>.jsonl`).
  Every usable file was copied into `../results/` in the canonical layout by
  `mdg/hpc/migrate_results.sh`, so these are duplicates.
- `per_recipe_csv/` — one CSV per dataset × model × recipe, plus the old
  `consolidated.csv`. Replaced by the single appended
  `../attack_results.csv` and `mdg/results/consolidated.csv`.

Safe to delete once you have confirmed the migrated results are complete
(`bash mdg/hpc/status.sh`).
