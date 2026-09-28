#!/usr/bin/env bash

set -euo pipefail

# Example commands for the forecasting workflows.
# Each command streams stdout/stderr to a timestamped file under logs/.

mkdir -p logs

# 1. Run inference without memory.
python test/run_forecast.py \
  --model gpt-5-mini \
  --search-provider tavily \
  --first-k 20 \
  --selected-week-idx 0 \
  --max-concurrency 8 \
  --max-search-calls 20 \
  --dataset prophet_arena \
  2>&1 | tee "logs/inference_with_filter_$(date +%Y%m%d_%H%M%S).log"


# 1a. Run inference without memory and disable close-time date filtering.
python test/run_forecast.py \
  --model gpt-5-mini \
  --search-provider serper \
  --selected-week-idx 7 \
  --dataset prophet_arena \
  --max-concurrency 6 \
  --max-search-calls 30 \
  --filter-days-before-close 7 \
  2>&1 | tee "logs/inference_with_filter_$(date +%Y%m%d_%H%M%S).log"



python test/run_forecast.py \
  --model gpt-5-mini \
  --search-provider serper \
  --selected-week-idx 0 \
  --dataset futurex_online \
  --max-concurrency 2 \
  --max-search-calls 30 \
  --filter-days-before-close 1 \
  2>&1 | tee "logs/inference_with_filter_$(date +%Y%m%d_%H%M%S).log"


python test/run_forecast.py \
  --model gemini-2.5-flash \
  --search-provider tavily \
  --selected-week-idx 4 \
  --dataset prophet_arena \
  --max-concurrency 2 \
  --max-search-calls 30 \
  --filter-days-before-close 7 \
  --resume-results-path results/prophet_arena/gemini-2.5-flash/tavily/week7/results_with_filter.jsonl \
  2>&1 | tee "logs/inference_no_filter_$(date +%Y%m%d_%H%M%S).log"


python test/run_forecast.py \
  --model gpt-5-mini \
  --search-provider serper \
  --selected-week-idx 4 \
  --no-close-date-filter \
  --max-search-calls 30 \
  --max-concurrency 5 \
  2>&1 | tee "logs/inference_no_filter_$(date +%Y%m%d_%H%M%S).log"

# 2. Train memory from a baseline results file.
python train/train_memory.py \
  --baseline-results-path results/prophet_arena/gpt-5-mini/serper/week7/baselines/results_with_filter.jsonl \
  --skip-classification \
  --model gpt-5-mini \
  --search-provider serper \
  --max-search-calls 30 \
  --max-concurrency 16 \
  --start-epoch 3 \
  --epochs 4 \
  2>&1 | tee "logs/train_memory_week7_$(date +%Y%m%d_%H%M%S).log"

python train/train_memory.py \
  --baseline-results-path results/prophet_arena/gpt-5-mini/serper/week7/baselines/results_with_filter.jsonl \
  --model gpt-5-mini \
  --search-provider serper \
  --max-search-calls 30 \
  --update-classification \
  --max-concurrency 6 \
  --start-epoch 4 \
  --epochs 5 \
  --provider openai \
  --taxonomy-path results/prophet_arena/gpt-5-mini/serper/week6/prophet_arena_ctgr.json \
  2>&1 | tee "logs/train_memory_week7_$(date +%Y%m%d_%H%M%S).log"


MODEL=gpt-5-mini
TOOL=serper
DATASET=futurex
WEEK=12
python train/train_memory.py \
  --baseline-results-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/results_with_filter.jsonl \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 2 \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/${DATASET}_ctgr.json \
  --initial-memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/memory_epochs/epoch_2/memory.json \
  --epochs 3 \
  --filter-days-before-close 7 \
  --start-epoch 2 \
  --dataset ${DATASET} \
  --provider azure \
  2>&1 | tee "logs/train_memory_week${WEEK}_${DATASET}_$(date +%Y%m%d_%H%M%S).log"


MODEL=gpt-5-mini
TOOL=serper
DATASET=prophet_arena
WEEK=6
nohup python train/train_memory.py \
  --baseline-results-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/results_with_filter.jsonl \
  --model ${MODEL} \
  --update-classification \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 6 \
  --start-epoch 4 \
  --epochs 5 \
  --filter-days-before-close 7 \
  --provider azure \
  --dataset ${DATASET} \
  2>&1 | tee "logs/train_memory_week${WEEK}_${DATASET}_$(date +%Y%m%d_%H%M%S).log"


# 2a. Train memory without close-time date filtering.
DATASET=prophet_arena
MODEL=gpt-5-mini
TOOL=serper
WEEK=10
EPOCH=3
nohup python train/train_memory.py \
  --baseline-results-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/results_with_filter.jsonl \
  --update-classification \
  --initial-memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --epochs 4 \
  --max-concurrency 4 \
  --dataset ${DATASET} \
  --provider azure \
  --max-search-calls 30 \
  --filter-days-before-close 7 \
  2>&1 | tee "logs/train_memory_week${WEEK}_filter_$(date +%Y%m%d_%H%M%S).log"


DATASET=futurex
MODEL=gpt-5-mini
TOOL=serper
WEEK=8
EPOCH=2
nohup python train/train_memory.py \
  --baseline-results-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/results_with_filter.jsonl \
  --update-classification \
  --initial-memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --start-epoch 5 \
  --epochs 5 \
  --max-concurrency 2 \
  --dataset ${DATASET} \
  --provider azure \
  --max-search-calls 30 \
  --filter-days-before-close 7 \
  2>&1 | tee "logs/train_memory_week${WEEK}_filter_$(date +%Y%m%d_%H%M%S).log"



DATASET=prophet_arena
MODEL=gemini-2.5-flash
TOOL=serper
WEEK=9
EPOCH=3
nohup python train/train_memory.py \
  --baseline-results-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/results_with_filter.jsonl \
  --update-classification \
  --initial-memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((${WEEK}-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --epochs 4 \
  --max-concurrency 6 \
  --dataset ${DATASET} \
  --provider gemini \
  --max-search-calls 30 \
  --filter-days-before-close 7 \
  2>&1 | tee "logs/train_memory_week${WEEK}_filter_$(date +%Y%m%d_%H%M%S).log"


DATASET=futurex
nohup python train/train_memory.py \
  --baseline-results-path results/${DATASET}/gpt-5-mini/serper/week8/baselines/results_with_filter.jsonl \
  --model gpt-5-mini \
  --search-provider serper \
  --start-epoch 4 \
  --epochs 4 \
  --update-classification \
  --max-search-calls 30 \
  --max-concurrency 3 \
  --dataset ${DATASET} \
  --provider azure \
  --filter-days-before-close 7 \
  2>&1 | tee "logs/train_memory_week8_revise_$(date +%Y%m%d_%H%M%S).log"



# 3. Run with provided memory and evaluate performance.
WEEK=11
TOOL=serper
MODEL=gemini-2.5-flash
DATASET=futurex
python test/run_forecast.py --memory-mode trained \
  --memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/memory_epochs/epoch_2/memory.json \
  --evaluated-data-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/evaluation_with_filter.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 6 \
  --provider gemini \
  --filter-days-before-close 7 \
  --dataset ${DATASET} \
  2>&1 | tee "logs/test_performance_week${WEEK}_$(date +%Y%m%d_%H%M%S).log"



WEEK=7
TOOL=serper
MODEL=gpt-5-mini
DATASET=prophet_arena
EPOCH=4
python test/run_forecast.py --memory-mode trained \
  --memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --evaluated-data-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/evaluation_with_filter.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 8 \
  --provider azure \
  --filter-days-before-close 7 \
  --dataset ${DATASET} \
  2>&1 | tee "logs/test_performance_week${WEEK}_$(date +%Y%m%d_%H%M%S).log"




WEEK=9
TOOL=serper
MODEL=gpt-5-mini
DATASET=futurex
EPOCH=4
nohup python test/run_forecast.py --memory-mode trained \
  --memory-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --evaluated-data-path results/${DATASET}/${MODEL}/${TOOL}/week${WEEK}/baselines/evaluation_with_filter.json \
  --taxonomy-path results/${DATASET}/${MODEL}/${TOOL}/week$((WEEK-1))/${DATASET}_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 3 \
  --filter-days-before-close 7 \
  --dataset ${DATASET} \
  2>&1 | tee "logs/test_performance_week${WEEK}_$(date +%Y%m%d_%H%M%S).log"



WEEK=13
TOOL=serper
MODEL=gpt-5-mini
EPOCH=2
python test/run_forecast.py --memory-mode trained \
  --memory-path results/futurex/${MODEL}/serper/week$((WEEK-1))/memory_epochs/epoch_${EPOCH}/memory.json \
  --taxonomy-path results/futurex/${MODEL}/${TOOL}/week$((WEEK-1))/futurex_ctgr.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 4 \
  --provider azure \
  --filter-days-before-close 7 \
  --dataset futurex_online \
  2>&1 | tee "logs/test_performance_week${WEEK}_$(date +%Y%m%d_%H%M%S).log"



# 3a. Run with provided memory, plus factor memory.
python test/run_forecast.py --memory-mode trained \
  --memory-path results/prophet_arena/${MODEL}/${TOOL}/week${WEEK}/memory.json \
  --evaluated-data-path results/prophet_arena/${MODEL}/${TOOL}/week$((WEEK-1))/baselines/evaluation.json \
  --taxonomy-path results/prophet_arena/${MODEL}/${TOOL}/week${WEEK}/forecasting_event_categories.json \
  --model ${MODEL} \
  --search-provider ${TOOL} \
  --max-search-calls 30 \
  --max-concurrency 16 \
  2>&1 | tee logs/test_week9_memory10_$(date +%Y%m%d_%H%M%S).log


# 3b. Run with provided memory and disable close-time date filtering.
python test/run_forecast.py --memory-mode trained \
  --memory-path results/prophet_arena/gpt-5-mini/tavily/week10/memory.json \
  --evaluated-data-path results/prophet_arena/gpt-5-mini/tavily/week10/evaluation.json \
  --model gpt-5-mini \
  --search-provider tavily \
  --first-k 20 \
  --no-close-date-filter \
  2>&1 | tee "logs/test_performance_week10_no_filter_$(date +%Y%m%d_%H%M%S).log"


# Cross-model transferability: Gemini-trained memory → GPT backbone (futurex week9-12)
for WEEK in 11; do
  TRAIN_WEEK=$((WEEK - 1))
  nohup python test/run_forecast.py --memory-mode trained \
    --memory-path         results/futurex/gemini-2.5-flash/serper/week${TRAIN_WEEK}/memory_epochs/epoch_2/memory.json \
    --evaluated-data-path results/futurex/gpt-5-mini/serper/week${WEEK}/baselines/evaluation_with_filter.json \
    --taxonomy-path       results/futurex/gemini-2.5-flash/serper/week${TRAIN_WEEK}/futurex_ctgr.json \
    --model               gpt-5-mini \
    --search-provider     serper \
    --max-search-calls    30 \
    --max-concurrency     4 \
    --provider            azure \
    --filter-days-before-close 7 \
    --dataset             futurex \
    --output-dir          results/futurex/gpt-5-mini/serper/week${WEEK}/memory_effectiveness/gemini_week${TRAIN_WEEK}_epoch2 \
    2>&1 | tee "logs/xmodel_gem${TRAIN_WEEK}_gpt${WEEK}_$(date +%Y%m%d_%H%M%S).log"
done
