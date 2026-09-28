import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.memory_pipeline as memory_pipeline
from utils.file_processor import get_weekly_csvs
from utils.agent_toolkit import parse_outcomes_arg
from utils.inference_runner import (
    run_custom_question,
    run_predictions,
    run_predictions_with_memory,
)


def _run_fresh(args: argparse.Namespace) -> None:
    """--memory-mode none|factor: run predictions on weekly data from scratch."""
    common_kwargs = {
        "selected_week_idx": args.selected_week_idx,
        "run_all_weeks": args.all_weeks,
        "model_name": args.model,
        "max_search_calls": args.max_search_calls if args.max_search_calls is not None else 20,
        "search_provider": args.search_provider or "serpapi",
        "use_existing_weekly_files": not args.collect_fresh,
        "weeks_limit": args.weeks_limit,
        "first_k": args.first_k,
        "filter_year": args.filter_year,
        "filter_days_before_close": (
            args.filter_days_before_close if args.filter_days_before_close is not None else 2
        ),
        "use_close_date_filter": not args.no_close_date_filter,
        "max_concurrency": args.max_concurrency,
        "resume_results_path": args.resume_results_path,
        "dataset": args.dataset,
        "provider": args.provider or "auto",
    }

    if args.memory_mode == "factor":
        asyncio.run(
            run_predictions_with_memory(
                **common_kwargs,
                initial_factor_memory_path=args.factor_memory_path,
                update_factor_memory=not args.freeze_memory_updates,
            )
        )
    else:
        asyncio.run(run_predictions(**common_kwargs))


def _run_trained(args: argparse.Namespace) -> None:
    """--memory-mode trained: evaluate a pre-trained subcategory memory against an existing baseline."""
    if args.memory_path is None:
        raise SystemExit("--memory-path is required when --memory-mode trained.")

    evaluated_data_path = args.evaluated_data_path
    if evaluated_data_path is None:
        weekly_csvs = get_weekly_csvs(dataset=args.dataset)
        idx = args.selected_week_idx or 0
        if idx >= len(weekly_csvs):
            raise ValueError(
                f"--selected-week-idx {idx} is out of range "
                f"(only {len(weekly_csvs)} CSV(s) found for dataset '{args.dataset}')."
            )
        evaluated_data_path = Path(weekly_csvs[idx])
        print(f"No --evaluated-data-path provided; using CSV: {evaluated_data_path}")

    taxonomy_path = args.taxonomy_path or memory_pipeline.get_default_taxonomy_path(args.dataset)

    common_kwargs = dict(
        memory_path=args.memory_path,
        evaluated_data_path=evaluated_data_path,
        classification_path=args.classification_path,
        taxonomy_path=taxonomy_path,
        output_dir=args.output_dir,
        model_name=args.model,
        max_search_calls=args.max_search_calls,
        max_concurrency=args.max_concurrency,
        search_provider=args.search_provider,
        filter_year=args.filter_year,
        filter_days_before_close=args.filter_days_before_close,
        use_close_date_filter=False if args.no_close_date_filter else None,
        factor_memory_path=args.factor_memory_path,
        first_k=args.first_k,
        skip_classification=args.skip_classification,
        overwrite_preprocess=args.overwrite_preprocess,
        provider=args.provider or "azure",
        resume_results_path=args.resume_results_path,
    )

    if args.ablations:
        summary = asyncio.run(memory_pipeline.run_memory_ablations(**common_kwargs))
        abl = summary.get("ablations", {})
        no_f = abl.get("no_factor_memory", {})
        no_r = abl.get("no_reasoning_memory", {})
        print(
            "Ablation runs finished.\n"
            f"Baseline avg Brier:         {summary.get('baseline_average_brier_score')}\n"
            f"Full memory avg Brier:      {summary.get('with_memory_average_brier_score')} (from existing summary)\n"
            f"[Ablation] No factor memory:    Brier={no_f.get('average_brier_score')}  "
            f"delta_vs_baseline={no_f.get('delta_brier_vs_baseline')}  "
            f"delta_vs_full={no_f.get('delta_brier_vs_full_memory')}\n"
            f"[Ablation] No reasoning memory: Brier={no_r.get('average_brier_score')}  "
            f"delta_vs_baseline={no_r.get('delta_brier_vs_baseline')}  "
            f"delta_vs_full={no_r.get('delta_brier_vs_full_memory')}\n"
            f"Saved summary: {summary.get('summary_output_path')}"
        )
    else:
        summary = asyncio.run(
            memory_pipeline.test_memory_effectiveness(
                **common_kwargs,
                final_output_path=args.final_output_path,
            )
        )
        print(
            "Memory effectiveness test finished.\n"
            f"Baseline avg Brier:    {summary['baseline_average_brier_score']}\n"
            f"With-memory avg Brier: {summary['with_memory_average_brier_score']}  "
            f"(delta: {summary['delta_brier_score']})\n"
            f"Saved summary: {summary['summary_output_path']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forecasting inference, optionally with memory."
    )

    # --- memory mode ---
    parser.add_argument(
        "--memory-mode",
        type=str,
        choices=["none", "factor", "trained"],
        default="none",
        help=(
            "none: plain baseline inference on weekly data. "
            "factor: run weekly data with incremental factor memory that updates as it goes. "
            "trained: evaluate a pre-trained subcategory memory (--memory-path) against an "
            "already-computed baseline (--evaluated-data-path)."
        ),
    )

    # --- shared args ---
    parser.add_argument(
        "--selected-week-idx",
        type=int,
        default=None,
        help="Index into the weekly CSV list. Defaults to 0 unless --all-weeks is set "
             "(fresh modes), or the newest week (trained mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to use for forecasting.",
    )
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=None,
        help="Maximum number of web search tool calls allowed. "
             "Defaults to 20 for none/factor mode, or the baseline setting for trained mode.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of forecasting tasks to run in parallel.",
    )
    parser.add_argument(
        "--search-provider",
        type=str,
        choices=["serpapi", "serper", "tavily"],
        default=None,
        help="Web search backend to use. "
             "Defaults to serpapi for none/factor mode, or the baseline setting for trained mode.",
    )
    parser.add_argument(
        "--filter-year",
        type=int,
        default=None,
        help="Restrict search results to a specific year when no close-time filter is available.",
    )
    parser.add_argument(
        "--filter-days-before-close",
        type=int,
        default=None,
        help="Set the search cutoff to midnight this many day(s) before each task's event timestamp. "
             "Defaults to 2 for none/factor mode, or the baseline setting for trained mode.",
    )
    parser.add_argument(
        "--no-close-date-filter",
        action="store_true",
        help="Disable the automatic search cutoff based on each task's close time.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["prophet_arena", "futurex", "futurex_online"],
        default="prophet_arena",
        help="Dataset to use (default: prophet_arena).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["azure", "openai", "auto", "gemini"],
        default=None,
        help="LLM provider. Defaults to auto for none/factor mode, or azure for trained mode.",
    )
    parser.add_argument(
        "--resume-results-path",
        type=str,
        default=None,
        help="Continue a run from an existing partially-completed results file.",
    )
    parser.add_argument(
        "--factor-memory-path",
        type=str,
        default=None,
        help="Load an existing factor_memory.json. Used by --memory-mode factor (as the initial "
             "memory to update) and --memory-mode trained (used alongside the subcategory memory).",
    )
    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        help="Only run the first k rows/tasks.",
    )

    # --- --memory-mode none|factor only (fresh weekly/custom run) ---
    parser.add_argument(
        "--all-weeks",
        action="store_true",
        help="Run one prediction for every available weekly CSV.",
    )
    parser.add_argument(
        "--weeks-limit",
        type=int,
        default=10,
        help="Maximum number of weekly CSV files to consider.",
    )
    parser.add_argument(
        "--collect-fresh",
        action="store_true",
        help="Collect weekly CSVs instead of using existing files.",
    )
    parser.add_argument(
        "--freeze-memory-updates",
        action="store_true",
        help="With --memory-mode factor: use retrieval memory during inference but do not update "
             "the memory with new tasks.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Run a custom question directly instead of loading weekly data "
             "(not supported with --memory-mode trained).",
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help='Possible outcomes for --question. Accepts JSON like \'["Yes","No"]\' or comma-separated text.',
    )

    # --- --memory-mode trained only ---
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=None,
        help="Path to the trained subcategory memory JSON. Required with --memory-mode trained.",
    )
    parser.add_argument(
        "--evaluated-data-path",
        type=Path,
        default=None,
        help="Path to an existing evaluation.json, results_with_filter.jsonl, results_no_filter.jsonl, "
             "or a weekly CSV. If omitted, --selected-week-idx is used to find a CSV from --dataset.",
    )
    parser.add_argument(
        "--classification-path",
        type=Path,
        default=None,
        help="Optional question_category_classification.json. If missing, it will be generated.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=None,
        help="Path to the forecasting taxonomy JSON. Defaults to init_ctgr/{dataset}_ctgr.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results and summary.",
    )
    parser.add_argument(
        "--final-output-path",
        type=Path,
        default=None,
        help="Optional explicit file path for the final effectiveness summary JSON "
             "(only used without --ablations).",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification and load from the existing cache file under the "
             "memory_effectiveness output directory. Errors if no cache exists.",
    )
    parser.add_argument(
        "--overwrite-preprocess",
        action="store_true",
        help="Regenerate classification labels even if they already exist.",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run ablation experiments only (no-factor-memory and no-reasoning-memory). "
             "Does NOT re-run the full memory test.",
    )

    args = parser.parse_args()

    if args.question:
        if args.memory_mode == "trained":
            parser.error("--question is not supported with --memory-mode trained.")
        asyncio.run(
            run_custom_question(
                question=args.question,
                markets=parse_outcomes_arg(args.outcomes),
                model_name=args.model,
                max_search_calls=args.max_search_calls if args.max_search_calls is not None else 20,
                search_provider=args.search_provider or "serpapi",
                filter_year=args.filter_year,
                filter_days_before_close=(
                    args.filter_days_before_close if args.filter_days_before_close is not None else 2
                ),
                use_close_date_filter=not args.no_close_date_filter,
            )
        )
        return

    if args.memory_mode == "trained":
        _run_trained(args)
    else:
        _run_fresh(args)


if __name__ == "__main__":
    main()
