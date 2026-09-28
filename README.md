# ForecastCompass

Official repository for the paper
[**ForecastCompass: Guiding Agentic Forecasting with Adaptive Factor Memory**](https://arxiv.org/abs/2605.30858), **Accepted at NeurIPS 2026**🎉.

A memory-augmented probabilistic forecasting pipeline. An LLM agent forecasts
outcomes for prediction-market questions (Prophet Arena, FutureX) week by
week, optionally guided by a "subcategory memory" that is trained on earlier
weeks and evaluated on later, held-out weeks.

The pipeline has three stages:

1. **Data process** — download raw prediction-market data and split it into
   weekly CSVs.
2. **Train** — train a subcategory memory across several epochs from a
   week's baseline results.
3. **Test** — run inference (with or without memory) and score the results.

## Setup

```bash
conda create -n forecasting python=3.10
conda activate forecasting

pip install -e .          # this project's dependencies
pip install -e src/       # vendored openai-agents SDK (src/agents)
```

Copy `.env.example` to `.env` and fill in the credentials for whichever
providers you use:

```bash
cp .env.example .env
```

## 1. Data process

Weekly CSVs live under `data/{dataset}/`. Prophet Arena is collected
automatically the first time it's needed, but FutureX and FutureX-Online must
be collected manually first:

```bash
# Prophet Arena — downloads prophetarena/Prophet-Arena-Subset-1200 from HuggingFace
python data_process/prophet_arena_loader.py --weeks-back 2 --duration 1

# FutureX — downloads futurex-ai/Futurex-Past
python data_process/futurex_loader.py

# FutureX-Online — downloads futurex-ai/Futurex-Online
python data_process/futurex_online_loader.py --week 12
```

`utils/results_io.get_weekly_csvs()` reads the resulting files; pass
`--collect-fresh` to `test/run_forecast.py` to re-collect Prophet Arena on the
fly instead of using cached weekly files.

## 2. Run without memory (baseline)

`test/run_forecast.py` is the single entrypoint for running inference; the
memory behavior is selected with `--memory-mode`. For a plain baseline run
(`--memory-mode none`, the default):

```bash
python test/run_forecast.py \
  --model gpt-5-mini \
  --search-provider serper \
  --selected-week-idx 0 \
  --dataset prophet_arena \
  --max-concurrency 8 \
  --max-search-calls 20 \
  --filter-days-before-close 7
```

This saves `results/{dataset}/{model}/{search_provider}/week{N}/results_with_filter.jsonl`,
which both step 3 (training) and step 4 (testing) build on. Add
`--memory-mode factor` to instead run with a lightweight factor memory that
updates incrementally as it processes the week's tasks.

## 3. Train memory

`train/train_memory.py` trains a subcategory memory across epochs from a
week's baseline results (`--baseline-results-path`). Per the paper's setup,
memory is trained for **3 epochs per week for Prophet Arena** and **2 epochs
per week for FutureX**:

```bash
# Prophet Arena: 3 epochs
python train/train_memory.py \
  --baseline-results-path results/prophet_arena/gpt-5-mini/serper/week6/baselines/results_with_filter.jsonl \
  --model gpt-5-mini \
  --search-provider serper \
  --update-classification \
  --dataset prophet_arena \
  --provider azure \
  --epochs 3

# FutureX: 2 epochs, carrying memory forward from the previous week
python train/train_memory.py \
  --baseline-results-path results/futurex/gpt-5-mini/serper/week8/baselines/results_with_filter.jsonl \
  --model gpt-5-mini \
  --search-provider serper \
  --dataset futurex \
  --provider azure \
  --taxonomy-path results/futurex/gpt-5-mini/serper/week7/futurex_ctgr.json \
  --initial-memory-path results/futurex/gpt-5-mini/serper/week7/memory_epochs/epoch_2/memory.json \
  --epochs 2
```

Each epoch's memory is saved to
`.../week{N}/memory_epochs/epoch_{k}/memory.json`, alongside that epoch's own
evaluation. Use `--start-epoch` to resume training partway through, and
`--initial-memory-path` / `--taxonomy-path` to carry a previous week's memory
and taxonomy forward into the next week.

## 4. Test memory

To evaluate a trained memory against a later, held-out week, use
`test/run_forecast.py --memory-mode trained`:

```bash
python test/run_forecast.py --memory-mode trained \
  --memory-path results/prophet_arena/gpt-5-mini/serper/week6/memory_epochs/epoch_3/memory.json \
  --evaluated-data-path results/prophet_arena/gpt-5-mini/serper/week7/baselines/evaluation_with_filter.json \
  --taxonomy-path results/prophet_arena/gpt-5-mini/serper/week6/prophet_arena_ctgr.json \
  --model gpt-5-mini \
  --search-provider serper \
  --provider azure \
  --dataset prophet_arena
```

This reports the baseline Brier score, the with-memory Brier score, and the
delta between them. Add `--ablations` to instead run the no-factor-memory /
no-reasoning-memory ablations against an existing summary.

To test one trained memory against several target weeks, call
`test/run_forecast.py --memory-mode trained` once per week with
`--evaluated-data-path` pointed at that week's baseline evaluation.

## Project layout

```
data_process/   HuggingFace dataset downloaders → weekly CSVs
train/          train_memory.py — per-epoch memory training
test/           run_forecast.py — inference + scoring
utils/          shared engine used by both train/ and test/
prompt/         prompt templates used by the memory pipeline
init_ctgr/      default taxonomy seeds (init_ctgr/{dataset}_ctgr.json)
src/agents/     vendored openai-agents SDK
```

## Citation

```bibtex
@misc{chang2026forecastcompassguidingagenticforecasting,
      title={ForecastCompass: Guiding Agentic Forecasting with Adaptive Factor Memory}, 
      author={Yurui Chang and Yongkang Du and Yuanpu Cao and Jinghui Chen and Lu Lin},
      year={2026},
      eprint={2605.30858},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.30858}, 
}
```
