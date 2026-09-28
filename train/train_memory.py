import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.memory_pipeline as memory_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the subcategory-memory forecasting pipeline across epochs."
    )
    parser.add_argument(
        "--baseline-results-path",
        type=Path,
        required=True,
        help="Path to the first-iteration results_with_filter.jsonl or results_no_filter.jsonl file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Total number of epochs to maintain, including the provided first iteration.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=None,
        help="Path to the forecasting taxonomy JSON. Defaults to init_ctgr/{dataset}_ctgr.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Forecasting model for epoch 2+ inference.",
    )
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=20,
        help="Maximum search calls during each forecasting run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of forecasting tasks to run in parallel during inference.",
    )
    parser.add_argument(
        "--search-provider",
        type=str,
        choices=["serpapi", "serper", "tavily"],
        default="serper",
        help="Search backend used during forecasting.",
    )
    parser.add_argument(
        "--filter-year",
        type=int,
        default=None,
        help="Optional fallback year filter when close-time filtering is unavailable.",
    )
    parser.add_argument(
        "--filter-days-before-close",
        type=int,
        default=2,
        help="Search cutoff applied at midnight this many day(s) before each task's event timestamp.",
    )
    parser.add_argument(
        "--no-close-date-filter",
        action="store_true",
        help="Disable the automatic search cutoff based on each task's close time.",
    )
    parser.add_argument(
        "--factor-memory-path",
        type=Path,
        default=None,
        help="Optional factor_memory.json to use alongside subcategory memory at inference time.",
    )
    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        help="Optionally limit inference runs to the first k tasks.",
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=8,
        help="Maximum number of common factors stored per subcategory memory.",
    )
    parser.add_argument(
        "--update-classification",
        action="store_true",
        help="Run classification once before memory training begins. By default classification is "
             "not updated and must be done separately before training.",
    )
    parser.add_argument(
        "--overwrite-preprocess",
        action="store_true",
        help="Re-run classification and taxonomy verification even if saved outputs already exist. "
             "Only applies when --update-classification is set.",
    )
    parser.add_argument(
        "--overwrite-epoch1-memory",
        action="store_true",
        help="Rebuild epoch-1 subcategory memory even if memory.json already exists.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Resume training from this epoch (e.g. 2 to skip epoch-1 memory init). "
             "Epoch-1 memory and prior epoch results must already exist on disk.",
    )
    parser.add_argument(
        "--initial-memory-path",
        type=Path,
        default=None,
        help="Path to an existing memory.json to seed epoch 1 from. "
             "When provided, epoch 1 revises the seeded memory instead of initialising from scratch. "
             "Use this to carry over memory from a previous week (e.g. week 10 memory into week 9 training).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["azure", "openai", "auto", "gemini"],
        default="azure",
        help="LLM provider to use: azure (default), openai, or auto (detect from env vars).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["prophet_arena", "futurex", "futurex_online"],
        default="prophet_arena",
        help="Dataset the baseline results were collected from (default: prophet_arena). "
             "Used for informational purposes; the dataset is implicit in the baseline-results-path.",
    )
    parser.add_argument(
        "--suggestion-batch-size",
        type=int,
        default=30,
        help="Number of per-event suggestions to merge into one batch summary before final memory revision (default: 30).",
    )
    args = parser.parse_args()

    taxonomy_path = args.taxonomy_path or memory_pipeline.get_default_taxonomy_path(args.dataset)

    pipeline_summary = asyncio.run(
        memory_pipeline.run_epoch_pipeline(
            baseline_results_path=args.baseline_results_path,
            epochs=args.epochs,
            taxonomy_path=taxonomy_path,
            dataset=args.dataset,
            model_name=args.model,
            max_search_calls=args.max_search_calls,
            max_concurrency=args.max_concurrency,
            search_provider=args.search_provider,
            filter_year=args.filter_year,
            filter_days_before_close=args.filter_days_before_close,
            use_close_date_filter=not args.no_close_date_filter,
            factor_memory_path=args.factor_memory_path,
            first_k=args.first_k,
            max_factors=args.max_factors,
            suggestion_batch_size=args.suggestion_batch_size,
            update_classification=args.update_classification or args.taxonomy_path is not None,
            overwrite_preprocess=args.overwrite_preprocess,
            overwrite_epoch1_memory=args.overwrite_epoch1_memory,
            start_epoch=args.start_epoch,
            initial_memory_path=args.initial_memory_path,
            provider=args.provider,
        )
    )

    print("Pipeline finished.")
    print(f"Latest memory: {pipeline_summary['latest_memory_path']}")
    epoch_summary_path = (
        Path(pipeline_summary["baseline_results_path"]).parent
        / "memory_epochs"
        / "pipeline_summary.json"
    )
    print(f"Epoch summary: {epoch_summary_path}")
    cost_summary = pipeline_summary.get("cost_summary") or {}
    if cost_summary:
        print(
            "Total LLM cost: "
            f"${cost_summary.get('estimated_cost_usd', 0.0):.4f} "
            f"({cost_summary.get('calls', 0)} calls, "
            f"{cost_summary.get('input_tokens', 0)} input tokens, "
            f"{cost_summary.get('output_tokens', 0)} output tokens)"
        )

    final_epoch = pipeline_summary["epochs"][-1]
    final_evaluation_path = Path(final_epoch["evaluation_path"])
    if final_evaluation_path.exists():
        with final_evaluation_path.open("r", encoding="utf-8") as f:
            final_evaluation = json.load(f)
        print(
            f"Final epoch average Brier score: "
            f"{final_evaluation.get('average_brier_score')}"
        )
        print(f"Final epoch evaluation: {final_evaluation_path}")


if __name__ == "__main__":
    main()
