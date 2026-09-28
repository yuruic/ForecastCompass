#!/usr/bin/env bash
# test_memory.sh
# Test a provided subcategory memory against each target week.
#
#   Prophet Arena : memory from week6/epoch_2  →  test on weeks 7-10
#   FutureX       : memory from week8/epoch_3  →  test on weeks 9-12
#
# Usage:
#   bash test/test_memory.sh [dataset] [provider] [model] [epoch]
#
#   bash test/test_memory.sh                                  # prophet_arena, azure, gpt-5-mini, epoch_2
#   bash test/test_memory.sh prophet_arena                    # same as above
#   bash test/test_memory.sh futurex                          # futurex, azure, gpt-5-mini, epoch_3
#   bash test/test_memory.sh prophet_arena azure gpt-5-mini 3 # use epoch_3 for prophet_arena
#   bash test/test_memory.sh futurex azure gpt-5-mini 2      # use epoch_2 for futurex
#   bash test/test_memory.sh prophet_arena gemini gemini-2.5-flash 3      # use epoch_2 for futurex

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── nohup self-relaunch ────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${_NOHUP_SELF:-}" ]; then
    mkdir -p logs
    LOGFILE="logs/test_memory_${1:-prophet_arena}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching in background → ${LOGFILE}"
    _NOHUP_SELF=1 nohup bash "$0" "$@" >"${LOGFILE}" 2>&1 &
    echo "PID: $!    tail -f ${LOGFILE}"
    exit 0
fi
# ──────────────────────────────────────────────────────────────────────────────

DATASET="${1:-prophet_arena}"
PROVIDER="${2:-azure}"
MODEL="${3:-gpt-5-mini}"

SEARCH=serper
MAX_SEARCH=30
CONCURRENCY=4
FILTER_DAYS=7

RESULTS_BASE="results/${DATASET}/${MODEL}/${SEARCH}"

# Memory source week, target test weeks, and default epoch per dataset.
if [ "${DATASET}" = "futurex" ]; then
    MEMORY_WEEK=week8
    DEFAULT_EPOCH=2
    TEST_WEEKS=(week10 week11 week12)
    # TEST_WEEKS=(week11 week12)
    # TEST_WEEKS=(week12)
else
    MEMORY_WEEK=week6
    DEFAULT_EPOCH=3
    TEST_WEEKS=(week8 week9 week10)
fi

EPOCH="${4:-${DEFAULT_EPOCH}}"

MEMORY_PATH="${RESULTS_BASE}/${MEMORY_WEEK}/memory_epochs/epoch_${EPOCH}/memory.json"
TAXONOMY_PATH="${RESULTS_BASE}/${MEMORY_WEEK}/${DATASET}_ctgr.json"

mkdir -p logs

echo "======================================================"
echo " Dataset    : ${DATASET}"
echo " Model      : ${MODEL}"
echo " Provider   : ${PROVIDER}"
echo " Memory     : ${MEMORY_PATH}"
echo " Taxonomy   : ${TAXONOMY_PATH}"
echo " Test weeks : ${TEST_WEEKS[*]}"
echo "======================================================"

if [ ! -f "${MEMORY_PATH}" ]; then
    echo "ERROR: memory file not found: ${MEMORY_PATH}" >&2
    exit 1
fi

if [ ! -f "${TAXONOMY_PATH}" ]; then
    echo "ERROR: taxonomy file not found: ${TAXONOMY_PATH}" >&2
    exit 1
fi

for WEEK in "${TEST_WEEKS[@]}"; do
    EVAL_PATH="${RESULTS_BASE}/${WEEK}/baselines/evaluation_with_filter.json"
    DONE_MARKER="${RESULTS_BASE}/${WEEK}/memory_effectiveness/${MEMORY_WEEK}_epoch_${EPOCH}/effectiveness_summary.json"

    if [ ! -f "${EVAL_PATH}" ]; then
        echo "WARNING: baseline evaluation not found, skipping ${WEEK}: ${EVAL_PATH}" >&2
        continue
    fi

    if [ -f "${DONE_MARKER}" ]; then
        echo ""
        echo "=== Skipping ${DATASET} ${WEEK} (already completed: ${DONE_MARKER}) ==="
        continue
    fi

    echo ""
    echo "=== Testing memory on ${DATASET} ${WEEK} ==="
    python test/run_forecast.py --memory-mode trained \
        --memory-path               "${MEMORY_PATH}" \
        --evaluated-data-path       "${EVAL_PATH}" \
        --taxonomy-path             "${TAXONOMY_PATH}" \
        --model                     "${MODEL}" \
        --search-provider           "${SEARCH}" \
        --max-search-calls          "${MAX_SEARCH}" \
        --max-concurrency           "${CONCURRENCY}" \
        --filter-days-before-close  "${FILTER_DAYS}" \
        --dataset                   "${DATASET}" \
        --provider                  "${PROVIDER}" \
        2>&1 | tee "logs/test_memory_${DATASET}_${WEEK}_epoch${EPOCH}_$(date +%Y%m%d_%H%M%S).log"

    if [ -f "${DONE_MARKER}" ]; then
        echo "=== Completed ${DATASET} ${WEEK} ==="
    else
        echo "WARNING: ${WEEK} finished but no summary found at ${DONE_MARKER}" >&2
    fi
done

echo ""
echo "======================================================"
echo " All done."
echo " Results : ${RESULTS_BASE}/week*/memory_effectiveness/${MEMORY_WEEK}_epoch_${EPOCH}/"
echo " Logs    : logs/test_memory_${DATASET}_*_epoch${EPOCH}_*.log"
echo "======================================================"
