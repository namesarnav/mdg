# vm-scripts

One self-contained script per VM. Bare Ubuntu box to running attacks, in one
command. Nothing needs to exist first except an HF token.

```bash
# on each VM, once
printf '%s' 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token

# VM1
bash vm1.sh
# VM2
bash vm2.sh
# VM3
bash vm3.sh
```

Each script is standalone — it clones the repo itself, so copy just the one
file to the box:

```bash
scp vm-scripts/vm1.sh user@vm1:~/
```

## What it does

1. apt: build deps, git, git-lfs, tmux, GNU parallel, rsync
2. pyenv + CPython (3.12.7, falling back to 3.11.9 if the lockfile won't solve)
3. Poetry, in-project `.venv`
4. clone the repo, `git lfs pull`, verify no file is still an LFS pointer
5. download all 33 models from the Hub, serially
6. warm the TextAttack asset cache, serially
7. launch three tmux sessions

## tmux sessions

| Session | What |
|---|---|
| `mdg-attack` | the run — `run_parallel.sh cells`, one worker per core |
| `mdg-backup` | copies every output to `~/mdg-archive` every 10 min, tarball hourly |
| `mdg-status` | progress table, refreshed every 5 min |

```bash
tmux attach -t mdg-attack     # Ctrl-B then D to detach
tmux ls
```

## Everything is saved

`~/mdg-archive/` receives, every 10 minutes:

- `mdg/adv_attack/results/**` — every `.jsonl` of per-example attack records,
  every `_summary.json`, every `all_summaries.json`
- `mdg/logs/**` — per-worker logs, one per cell and row shard
- `attack_results.csv`, `consolidated.csv`
- `vmN-YYYYMMDD-HHMM.tar.gz` — hourly snapshot of the whole lot
- `archive-errors.log` — any copy that failed. **Check this.** An empty file
  means every backup cycle succeeded.

Nothing is ever deleted: `MDG_KEEP_CHECKPOINTS=1` is set, and the archiver only
ever adds.

## Settings

Identical on all three VMs, overridable by environment variable:

| Variable | Default | |
|---|---|---|
| `ROW_SHARDS` | 64 | **must be identical on every VM** |
| `JOBS` | `nproc` | concurrent workers |

There is no example cap and no query budget. The run covers the **complete test
split** of every dataset with unlimited search depth, and both knobs are
explicitly `unset` inside the script so a stray value in your shell cannot cap
it.

`ROW_SHARDS` is baked into the output filenames (`<recipe>.sh3of8.jsonl`). A
cell attacked with a different count on one VM cannot be merged with the same
cell from another.

## Re-running

Safe. Every finished recipe writes a done-marker and is skipped, so a crashed
or evicted VM resumes by running the same script again.

## When all three finish

Collect onto one box and merge:

```bash
rsync -a vm1:~/mdg-archive/mdg/adv_attack/results/ mdg/adv_attack/results/
rsync -a vm2:~/mdg-archive/mdg/adv_attack/results/ mdg/adv_attack/results/
rsync -a vm3:~/mdg-archive/mdg/adv_attack/results/ mdg/adv_attack/results/

.venv/bin/python mdg/scripts/merge_shards.py
.venv/bin/python -m mdg.scripts.consolidate_results
```

Final table: `mdg/results/consolidated.csv`.
