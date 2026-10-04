#!/usr/bin/env bash
# Quick pre-flight: can I actually get a GPU on this cluster?
#
#   bash mdg/hpc/check_access.sh              # try the default partitions
#   bash mdg/hpc/check_access.sh a100 l40s    # try these, in order
#
# Asks for ONE GPU for 5 minutes and runs nvidia-smi. Prints PASS on the first
# partition that works. Safe to run from a login node — it submits, it does not
# compute. Run it before 00_setup_env.sh; if this fails, nothing else will work.
set -uo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"

echo "========================================================"
echo "  USER    : $USER        CLUSTER: $(hostname)"
echo "========================================================"

# ── 1. Which Slurm accounts do I belong to? ───────────────────────────────────
echo ""
echo "[1/3] Slurm accounts"
ACCTS=$(sacctmgr -nP show assoc user="$USER" format=account 2>/dev/null | sort -u | grep -v '^$')
if [ -z "$ACCTS" ]; then
  echo "  (none found — sacctmgr returned nothing)"
else
  echo "$ACCTS" | sed 's/^/    /'
fi

# Pick an account: explicit $ACCOUNT, else the first project-looking one.
if [ -z "${ACCOUNT:-}" ]; then
  ACCOUNT=$(echo "$ACCTS" | grep -v '^users$' | head -1)
fi
if [ -z "${ACCOUNT:-}" ]; then
  echo ""
  echo "  [BLOCKED] No project account. Every job will fail with"
  echo "            'Invalid Slurm account'. Ask your PI to register a project"
  echo "            at https://projects.hpc.nyu.edu, then:"
  echo "              echo torch_pr_xxx_yyy > ~/.slurm_account"
  echo "            (On a cluster that needs no account, ignore this.)"
else
  echo "  using account: $ACCOUNT"
fi
ACC_ARG=()
[ -n "${ACCOUNT:-}" ] && ACC_ARG=(--account="$ACCOUNT")

# ── 2. How busy are the candidate partitions? ─────────────────────────────────
# Default order is smallest-sufficient first: t5-base (220M) and Llama-3.2-1B
# need ~10-20GB, so a 24-48GB card is plenty. Asking for an H200/B200 only
# means a longer queue for no gain.
CANDIDATES=("$@")
[ ${#CANDIDATES[@]} -eq 0 ] && CANDIDATES=(l40s rtx6000 a100)

echo ""
echo "[2/3] Partition availability (A=allocated I=idle O=other T=total)"
for p in "${CANDIDATES[@]}"; do
  line=$(sinfo -h -s -p "$p" -o "%P %a %D %F" 2>/dev/null | head -1)
  if [ -z "$line" ]; then
    echo "    $p — does not exist here"
  else
    echo "    $line"
  fi
done

# ── 3. Actually try to get a GPU ──────────────────────────────────────────────
echo ""
echo "[3/3] Requesting 1 GPU for 5 minutes (Ctrl-C to stop waiting)"
for p in "${CANDIDATES[@]}"; do
  echo ""
  echo "  --- trying partition: $p ---"
  OUT=$(srun -p "$p" "${ACC_ARG[@]}" --gres=gpu:1 --cpus-per-task=2 --mem=8G \
          --time=00:05:00 --job-name=mdg-gputest \
          bash -c 'echo "node=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' 2>&1)
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    echo "$OUT" | sed 's/^/    /'
    echo ""
    echo "  [PASS] GPU allocation works on '$p'."
    echo "         Use it for everything:"
    echo "           PARTITION=$p${ACCOUNT:+ ACCOUNT=$ACCOUNT} bash mdg/hpc/submit_all.sh"
    exit 0
  fi
  echo "$OUT" | head -5 | sed 's/^/    /'
  echo "  [FAIL] $p"
done

echo ""
echo "  [BLOCKED] No partition gave a GPU. The first error above says why:"
echo "    'Invalid Slurm account'     → you need a project account (step 1)"
echo "    'Invalid partition'/'access denied' → try another: bash $0 l40s_public a100"
echo "    queued forever              → the cluster is full; leave it queued via sbatch"
exit 1
