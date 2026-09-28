from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import datetime
import re
from pathlib import Path
from typing import Any

import os

from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    set_default_openai_client,
    set_tracing_disabled,
)
import openai
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from utils.agent_run_parser import inspect_agent_result
from utils.cost_tracker import estimate_cost, record_cost
from utils.data_loader import create_prediction_prompt, extract_filter_date
from utils.factor_memory import (
    DEFAULT_FACTOR_MEMORY_CONFIG,
    FactorMemoryConfig,
    build_factor_memory_context,
    build_or_update_factor_memory,
    retrieve_factor_memory_from_decomposition,
)
from utils.agent_toolkit import (
    USE_EXISTING_WEEKLY_FILES,
    BudgetHooks,
    FactorDecompositionOutput,
    ForecastBudget,
    attach_decomposition_to_session_steps,
    build_custom_task,
    build_factor_memory_summary,
    build_search_tool,
    build_week_results_payload,
    clear_existing_output_files,
    extract_probabilities_from_answer,
    get_custom_output_dir,
    get_week_output_dir,
    get_weekly_csvs,
    load_factor_memory,
    load_results_payload,
    load_weekly_tasks,
    resolve_results_path,
    save_custom_result,
    save_factor_memory,
    save_factor_memory_history,
    save_week_results,
    trajectory_self_corrector_with_backbone,
)

MAX_EMPTY_OUTPUT_RETRIES = 3

# Route all SDK calls through the Azure client; disable tracing.
_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
if _azure_endpoint and _azure_api_key:
    _azure_client = AsyncOpenAI(
        api_key=_azure_api_key,
        base_url=f"{_azure_endpoint}/openai/v1/",
    )
    set_default_openai_client(_azure_client, use_for_tracing=False)
set_tracing_disabled(True)


def _make_inference_client(provider: str) -> AsyncOpenAI | None:
    """Return a configured AsyncOpenAI client for the given provider, or None for the default."""
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "openai":
        return AsyncOpenAI()
    if provider == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        return AsyncOpenAI(api_key=api_key, base_url=f"{endpoint}/openai/v1/")
    return None  # "auto" — keep whatever was set at module load


def _resolve_agent_model(model_name: str) -> str | OpenAIChatCompletionsModel:
    """Return a string model name for Azure/OpenAI, or an OpenAIChatCompletionsModel for Gemini.

    Gemini requires Chat Completions (not the Responses API), and its client must be
    passed explicitly to bypass the provider's Azure env-var check.
    """
    if model_name.startswith("gemini"):
        gemini_client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return OpenAIChatCompletionsModel(model=model_name, openai_client=gemini_client)
    return model_name


DEFAULT_MODEL_TEMPERATURE = 0.7
DEFAULT_MODEL_TOP_P = 0.7

# Models that support reasoning.effort and verbosity (gpt-5+ and o-series).
# gpt-4o, gpt-4o-mini, gpt-4.1-* do NOT support these parameters.
def _model_supports_reasoning(model_name: str) -> bool:
    return model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _model_temperature(model_name: str) -> float | None:
    if _model_supports_reasoning(model_name):
        return None
    return DEFAULT_MODEL_TEMPERATURE


def _model_top_p(model_name: str) -> float | None:
    if _model_supports_reasoning(model_name):
        return None
    return DEFAULT_MODEL_TOP_P


def _print_usage(label: str, input_tokens: int, output_tokens: int, model_name: str) -> None:
    cost = estimate_cost(model_name, input_tokens, output_tokens)
    if label != "total":
        record_cost(f"inference:{label}", input_tokens, output_tokens, cost)
    cost_str = f"  est. cost ${cost:.4f}" if cost is not None else ""
    print(
        f"[{label}] tokens: {input_tokens} in / {output_tokens} out{cost_str}"
    )


def _partial_token_usage(s1_usage: Any, s2_usage: Any) -> dict[str, int | None]:
    """Build a token_usage dict for error paths where session2 may not have run."""
    s1_in = s1_usage.input_tokens if s1_usage is not None else None
    s1_out = s1_usage.output_tokens if s1_usage is not None else None
    s2_in = s2_usage.input_tokens if s2_usage is not None else None
    s2_out = s2_usage.output_tokens if s2_usage is not None else None
    return {
        "session1_input_tokens": s1_in,
        "session1_output_tokens": s1_out,
        "session2_input_tokens": s2_in,
        "session2_output_tokens": s2_out,
        "total_input_tokens": (s1_in or 0) + (s2_in or 0) if (s1_in is not None or s2_in is not None) else None,
        "total_output_tokens": (s1_out or 0) + (s2_out or 0) if (s1_out is not None or s2_out is not None) else None,
    }


def _sample_needs_rerun(sample: dict[str, Any]) -> bool:
    """Treat saved error-like samples as unfinished during resume."""
    status = str(sample.get("status") or "").strip().lower()
    if sample.get("error"):
        return True
    return status in {"error", "max_turns_exceeded", "no_output"}


async def run_predictions_with_memory(
    selected_week_idx: int | None = None,
    run_all_weeks: bool = False,
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 10,
    search_provider: str = "serpapi",
    use_existing_weekly_files: bool = USE_EXISTING_WEEKLY_FILES,
    weeks_limit: int = 10,
    first_k: int | None = None,
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
    factor_memory_config: FactorMemoryConfig | None = None,
    initial_factor_memory: dict[str, Any] | None = None,
    initial_factor_memory_path: str | None = None,
    update_factor_memory: bool = True,
    resume_results_path: str | None = None,
    max_concurrency: int = 5,
    dataset: str = "prophet_arena",
    provider: str = "auto",
) -> list[dict]:
    """Run predictions with retrieval memory and optional memory updates."""
    del max_concurrency
    client = _make_inference_client(provider)
    if client is not None:
        set_default_openai_client(client, use_for_tracing=False)
    max_turns = max_search_calls
    if resume_results_path is not None and run_all_weeks:
        raise ValueError("--resume-results-path cannot be combined with --all-weeks.")

    if resume_results_path is not None:
        saved_results_payload = load_results_payload(resume_results_path)
        weekly_csvs = [str(saved_results_payload["selected_week_csv"])]
        week_indexes = [int(saved_results_payload.get("week_index", 0))]
    else:
        weekly_csvs = get_weekly_csvs(
            use_existing_weekly_files=use_existing_weekly_files,
            weeks_limit=weeks_limit,
            dataset=dataset,
        )

        if run_all_weeks:
            week_indexes = list(range(len(weekly_csvs)))
        else:
            week_indexes = [0 if selected_week_idx is None else selected_week_idx]

    results = []
    if initial_factor_memory is not None and initial_factor_memory_path is not None:
        raise ValueError("Provide either initial_factor_memory or initial_factor_memory_path, not both.")

    if initial_factor_memory_path is not None:
        factor_memory = load_factor_memory(initial_factor_memory_path)
    elif initial_factor_memory is not None:
        factor_memory = copy.deepcopy(initial_factor_memory)
    else:
        factor_memory = None

    factor_memory_config = factor_memory_config or DEFAULT_FACTOR_MEMORY_CONFIG

    async def trajectory_self_corrector(prediction_result: Any) -> dict[str, Any]:
        return await trajectory_self_corrector_with_backbone(
            prediction_result,
            model_name=model_name,
        )

    async def rebuild_memory_state(
        saved_samples: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        rebuilt_memory = copy.deepcopy(factor_memory)
        rebuilt_history_entries: list[dict[str, Any]] = []
        for completed_idx, completed_result in enumerate(saved_samples):
            if update_factor_memory:
                rebuilt_memory = await build_or_update_factor_memory(
                    prediction_results=[completed_result],
                    existing_memory=rebuilt_memory,
                    config=factor_memory_config,
                    trajectory_self_corrector=trajectory_self_corrector,
                )
            rebuilt_history_entries.append(
                {
                    "task_idx": completed_idx,
                    "question": completed_result.get("question"),
                    "memory_updated": update_factor_memory,
                    "factor_memory_summary": (
                        build_factor_memory_summary(rebuilt_memory)
                        if rebuilt_memory is not None
                        else {}
                    ),
                    "factor_memory": copy.deepcopy(rebuilt_memory),
                }
            )
        return rebuilt_memory, rebuilt_history_entries

    for loop_idx, week_idx in enumerate(week_indexes):
        if resume_results_path is not None:
            selected_week_csv = weekly_csvs[loop_idx]
        else:
            selected_week_csv = weekly_csvs[week_idx]
        print()
        print(f"=== Running week index {week_idx}: {selected_week_csv} ===")

        weekly_tasks = load_weekly_tasks(selected_week_csv, first_k=first_k)
        if not weekly_tasks:
            raise ValueError(
                f"No tasks are available in {selected_week_csv}."
            )

        week_output_dir = get_week_output_dir(
            selected_week_csv,
            model_name=model_name,
            search_provider=search_provider,
            use_close_date_filter=use_close_date_filter,
            dataset=dataset,
        )
        sample_results: list[dict] = []
        if resume_results_path is not None:
            results_path = Path(resume_results_path)
            if results_path.parent != week_output_dir:
                week_output_dir = results_path.parent
        else:
            results_path = resolve_results_path(week_output_dir, use_close_date_filter)
        factor_memory_path = week_output_dir / "factor_memory.json"
        factor_memory_history_path = week_output_dir / "factor_memory_history.json"
        factor_memory_history_entries: list[dict[str, Any]] = []
        start_task_idx = 0
        pending_rerun_indices: list[int] = []

        if resume_results_path is not None:
            existing_results_payload = load_results_payload(results_path)
            saved_max_search_calls = existing_results_payload.get("max_search_calls")
            sample_results = copy.deepcopy(existing_results_payload.get("samples", []))
            start_task_idx = len(sample_results)
            if start_task_idx > len(weekly_tasks):
                raise ValueError(
                    f"Saved results contain {start_task_idx} samples, but only {len(weekly_tasks)} tasks are available."
                )
            pending_rerun_indices = [
                idx
                for idx, saved_sample in enumerate(sample_results)
                if _sample_needs_rerun(saved_sample)
            ]
            if (
                saved_max_search_calls is not None
                and int(saved_max_search_calls) != int(max_search_calls)
            ):
                print(
                    f"Saved results use max_search_calls={saved_max_search_calls}, "
                    f"but the current run uses {max_search_calls}. "
                    "Keeping successful samples and re-running saved error samples only."
                )
            print(
                f"Resuming from {results_path} with {start_task_idx} saved task(s), "
                f"{len(pending_rerun_indices)} needing rerun."
            )
        else:
            clear_existing_output_files(
                results_path,
                factor_memory_path,
                factor_memory_history_path,
            )

        if resume_results_path is not None and pending_rerun_indices:
            for task_idx in pending_rerun_indices:
                print(f"Re-running saved error sample {task_idx} before continuing.")
                prediction_result = await run_single_prediction(
                    sample_task=weekly_tasks[task_idx],
                    selected_week_csv=selected_week_csv,
                    task_idx=task_idx,
                    model_name=model_name,
                    max_search_calls=max_search_calls,
                    search_provider=search_provider,
                    filter_year=filter_year,
                    filter_days_before_close=filter_days_before_close,
                    use_close_date_filter=use_close_date_filter,
                    factor_memory=factor_memory,
                    use_factor_memory=True,
                )
                sample_results[task_idx] = prediction_result["data"]

            factor_memory, factor_memory_history_entries = await rebuild_memory_state(sample_results)
            week_results_payload = build_week_results_payload(
                selected_week_csv=selected_week_csv,
                week_idx=week_idx,
                model_name=model_name,
                max_search_calls=max_search_calls,
                search_provider=search_provider,
                filter_year=filter_year,
                filter_days_before_close=filter_days_before_close,
                use_close_date_filter=use_close_date_filter,
                sample_results=sample_results,
            )
            results_path = save_week_results(
                week_results_payload,
                week_output_dir,
                use_close_date_filter=use_close_date_filter,
            )
            factor_memory_history_payload = {
                "selected_week_csv": selected_week_csv,
                "week_index": week_idx,
                "search_provider": search_provider,
                "model": model_name,
                "factor_memory_updates_enabled": update_factor_memory,
                "initial_factor_memory_path": initial_factor_memory_path,
                "num_tasks_recorded": len(factor_memory_history_entries),
                "tasks": factor_memory_history_entries,
            }
            factor_memory_history_path = save_factor_memory_history(
                factor_memory_history_payload,
                week_output_dir,
            )
            if factor_memory is not None:
                factor_memory_path = save_factor_memory(factor_memory, week_output_dir)
        elif resume_results_path is not None and sample_results:
            factor_memory, factor_memory_history_entries = await rebuild_memory_state(sample_results)
            if factor_memory is not None:
                factor_memory_path = save_factor_memory(factor_memory, week_output_dir)

        for task_idx, sample_task in enumerate(weekly_tasks[start_task_idx:], start=start_task_idx):
            prediction_result = await run_single_prediction(
                sample_task=sample_task,
                selected_week_csv=selected_week_csv,
                task_idx=task_idx,
                model_name=model_name,
                max_search_calls=max_search_calls,
                search_provider=search_provider,
                filter_year=filter_year,
                filter_days_before_close=filter_days_before_close,
                use_close_date_filter=use_close_date_filter,
                factor_memory=factor_memory,
                use_factor_memory=True,
            )
            sample_results.append(prediction_result["data"])
            if update_factor_memory:
                factor_memory = await build_or_update_factor_memory(
                    prediction_results=[prediction_result["data"]],
                    existing_memory=factor_memory,
                    config=factor_memory_config,
                    trajectory_self_corrector=trajectory_self_corrector,
                )
            factor_memory_history_entries.append(
                {
                    "task_idx": task_idx,
                    "question": prediction_result["data"].get("question"),
                    "memory_updated": update_factor_memory,
                    "factor_memory_summary": (
                        build_factor_memory_summary(factor_memory)
                        if factor_memory is not None
                        else {}
                    ),
                    "factor_memory": copy.deepcopy(factor_memory),
                }
            )
            factor_memory_history_payload = {
                "selected_week_csv": selected_week_csv,
                "week_index": week_idx,
                "search_provider": search_provider,
                "model": model_name,
                "factor_memory_updates_enabled": update_factor_memory,
                "initial_factor_memory_path": initial_factor_memory_path,
                "num_tasks_recorded": len(factor_memory_history_entries),
                "tasks": factor_memory_history_entries,
            }
            factor_memory_history_path = save_factor_memory_history(
                factor_memory_history_payload,
                week_output_dir,
            )
            week_results_payload = build_week_results_payload(
                selected_week_csv=selected_week_csv,
                week_idx=week_idx,
                model_name=model_name,
                max_search_calls=max_search_calls,
                search_provider=search_provider,
                filter_year=filter_year,
                filter_days_before_close=filter_days_before_close,
                use_close_date_filter=use_close_date_filter,
                sample_results=sample_results,
            )
            results_path = save_week_results(
                week_results_payload,
                week_output_dir,
                use_close_date_filter=use_close_date_filter,
            )

        factor_memory_summary = (
            build_factor_memory_summary(factor_memory) if factor_memory is not None else {}
        )
        if factor_memory is not None:
            factor_memory_path = save_factor_memory(factor_memory, week_output_dir)
        else:
            factor_memory_path = week_output_dir / "factor_memory.json"

        results.append(
            {
                "selected_week_csv": selected_week_csv,
                "week_index": week_idx,
                "week_output_dir": str(week_output_dir),
                "num_samples": len(sample_results),
                "search_provider": search_provider,
                "filter_year": filter_year,
                "filter_days_before_close": filter_days_before_close,
                "use_close_date_filter": use_close_date_filter,
                "samples": sample_results,
                "factor_memory_file": str(factor_memory_path) if factor_memory is not None else None,
                "factor_memory_history_file": str(factor_memory_history_path),
                "factor_memory_summary": factor_memory_summary,
                "factor_memory_updates_enabled": update_factor_memory,
            }
        )

    print()
    print("=== Summary ===")
    for idx, result in zip(week_indexes, results):
        factor_memory_summary = result.get("factor_memory_summary", {})
        print(
            f"week index {idx}: {Path(result['selected_week_csv']).name} "
            f"-> {result['num_samples']} samples saved, "
            f"factor slots: {factor_memory_summary.get('num_factor_slots', 0)}"
        )

    return results


async def run_single_prediction(
    sample_task: dict,
    selected_week_csv: str,
    task_idx: int,
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 10,
    search_provider: str = "serpapi",
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
    factor_memory: dict[str, Any] | None = None,
    use_factor_memory: bool = False,
    include_weak_factor_memory: bool = False,
    external_memory_factor: str | None = None,
    external_memory_factor_raw: dict[str, Any] | None = None,
    external_memory_reasoning: str | None = None,
    external_memory_reasoning_raw: dict[str, Any] | None = None,
    external_memory_metadata: dict[str, Any] | None = None,
) -> dict:
    """Run one forecast for a single task."""
    max_turns = max_search_calls
    instruction, question = create_prediction_prompt(sample_task)
    filter_date_max = None
    if use_close_date_filter and sample_task.get("close_time") is not None:
        filter_date_max = extract_filter_date(
            sample_task["close_time"],
            days_before=filter_days_before_close,
        )

    print()
    print(
        f"Running sample {task_idx} from {Path(selected_week_csv).name} "
        f"with model {model_name} using {search_provider}"
    )
    if filter_date_max is not None:
        print(
            "Search results filtered to sources on or before "
            f"{filter_date_max} (midnight cutoff {filter_days_before_close} day(s) before event time)"
        )
    elif use_close_date_filter:
        print("Close-time date filtering enabled, but no close time was available for this task.")
    elif filter_year is not None:
        print(f"Search results filtered to year {filter_year}")
    else:
        print("Search results are not constrained by close-time date filtering.")
    print(question)

    started_at = datetime.now().isoformat()
    start_time = time.perf_counter()
    budget = ForecastBudget(max_search_calls=max_search_calls, task_idx=task_idx)
    session = SQLiteSession(session_id=f"forecast-{task_idx}-{int(time.time() * 1000)}")
    search_tool = build_search_tool(
        search_provider=search_provider,
        filter_year=filter_year,
        filter_date_max=filter_date_max,
    )
    retrieved_factor_memory: list[dict[str, Any]] = []
    retrieved_factor_memory_keys: list[str] = []
    factor_memory_context = ""
    if use_factor_memory and factor_memory is not None:
        retrieved_factor_memory = await retrieve_factor_memory_from_decomposition(
            factor_memory=factor_memory,
            decomposition=sample_task.get("title") or question,
            include_weak=include_weak_factor_memory,
        )
        retrieved_factor_memory_keys = [
            str(slot.get("factor_key"))
            for slot in retrieved_factor_memory
            if slot.get("factor_key")
        ]
        factor_memory_context = build_factor_memory_context(retrieved_factor_memory)
        print(
            "retrieved factor memory:",
            ", ".join(retrieved_factor_memory_keys) if retrieved_factor_memory_keys else "none",
        )

    decomposition_memory_instruction = ""
    if external_memory_factor:
        decomposition_memory_instruction = (
            "\n\nRelevant factor memory for this question's subcategory is provided below."
            "\nUse the listed factors as the primary starting structure for your decomposition."
            "\nFor each factor, incorporate the listed common reasoning patterns and actively avoid the listed common failures."
            "\nYou may add or drop factors only if there is clear evidence this specific question warrants it."
            f"\n\n{external_memory_factor}"
        )
    elif factor_memory_context:
        decomposition_memory_instruction = (
            "\n\nRetrieved factor memory for this question's subcategory is provided below."
            "\nUse the listed factors as the primary starting structure for your decomposition."
            "\nFor each factor, incorporate the listed common reasoning patterns and actively avoid the listed common failures."
            "\nYou may add or drop factors only if there is clear evidence this specific question warrants it."
            f"\n\n{factor_memory_context}"
        )
    decomposition_input = question + decomposition_memory_instruction
    decomposition_agent = Agent(
        name="ForecastDecompositionAgent",
        model=_resolve_agent_model(model_name),
        instructions=instruction + (
            "\nThis is session 1."
            "\nDecompose the forecasting task into the key factors you would investigate. The max number of factors is 5, but use fewer if that seems sufficient. Focus on the most important factors that will drive the forecast."
            "\nIf factor memory is provided, treat it as weak prior structure rather than as an authoritative template."
            "\nUse retrieved factor memory only to suggest candidate factors that may be relevant."
            "\nYou may discard retrieved factors if they do not fit the current case."
            "\nDo not simply mirror the retrieved factor list; adapt it to the current question."
            "\nFor each decomposed factor, also generate a short statement of its typical effect on the forecast output or probabilities."
            "\nFor each decomposed factor, also generate a short possible error or reasoning trap that should be avoided when using that factor."
            "\nIf factor memory is provided, use it as background context, but do not copy or manually select an error pattern from it. Infer the most relevant error-to-avoid yourself."
            "\nFactor memory may help rank or surface plausible drivers, but it must not by itself determine the final decomposition or imply strong confidence."
            "\nDo not use any tools."
            "\nDo not give a final answer or probabilities yet."
            "\nReturn only valid JSON with a top-level key 'factors'."
            "\n'factors' must be a list of objects with keys 'factor_name', 'rationale', 'typical_effect_on_output', and 'potential_common_error_pattern'."
            "\nSet 'typical_effect_on_output' to a concise description of how the factor usually shifts probabilities or changes the forecast."
            "\nSet 'potential_common_error_pattern' to a concise possible error to avoid for that factor, or an empty string only if none is relevant."
            "\nUse short reusable factor names rather than full-sentence descriptions."
        ),
        tools=[],
        output_type=FactorDecompositionOutput,
        model_settings=ModelSettings(
            parallel_tool_calls=False,
            temperature=_model_temperature(model_name),
            top_p=_model_top_p(model_name),
            reasoning=Reasoning(effort="low", summary="detailed") if _model_supports_reasoning(model_name) else None,
            verbosity="medium" if _model_supports_reasoning(model_name) else None,
        ),
    )
    factor_decomposition = None
    s1_usage = None
    s2_usage = None

    try:
        decomposition_result = await Runner.run(
            decomposition_agent,
            decomposition_input,
            session=session,
            max_turns=1,
        )
        factor_decomposition = decomposition_result.final_output.model_dump()
        print("factor decomposition:")
        print(json.dumps(factor_decomposition, indent=2, ensure_ascii=True))
        s1_usage = decomposition_result.context_wrapper.usage
        _print_usage("session1", s1_usage.input_tokens, s1_usage.output_tokens, model_name)

        forecast_agent = Agent(
            name="ForecastAnswerAgent",
            model=_resolve_agent_model(model_name),
            instructions=instruction + (
                "\nThis is session 2."
                "\nUse the prior factor decomposition from the conversation as your search plan."
                "\nIf reasoning memory is provided, use it as a reusable prior rather than as ground truth."
                "\nTreat the retrieved reasoning patterns as explicit guardrails for how to reason about the task."
                "\nYou should actively use them to avoid the listed common errors, distribution mistakes, calibration mistakes, and other reasoning traps."
                f"\nYou CAN ONLY use web search at most {max_search_calls} times total."
                f"\nYou MUST finish the task within {max_turns} turns total for this session."
                "\nBudget your turns carefully and keep enough remaining turns to produce the final answer."
                "\nIf the evidence is already sufficient, stop searching early rather than risking a max-turns failure."
                "\nDo not spend your last available turn on search or extra reasoning; reserve it for the final JSON answer."
                "\nDo not use all searches unless necessary."
                "\nOnce you have enough evidence, stop searching and return the final JSON immediately."
                "\nIf reasoning memory is provided in the user input, use it as a checklist for reasoning and evidence gathering, and explicitly check your forecast against it before answering."
            ),
            tools=[search_tool],
            model_settings=ModelSettings(
                parallel_tool_calls=False,
                temperature=_model_temperature(model_name),
                top_p=_model_top_p(model_name),
                reasoning=Reasoning(effort="medium", summary="detailed") if _model_supports_reasoning(model_name) else None,
                verbosity="high" if _model_supports_reasoning(model_name) else None,
            ),
        )
        second_session_input = (
            "Continue from the factor decomposition above."
            " You may now use web search if needed to answer the original forecasting question."
        )
        reasoning_memory_text = external_memory_reasoning
        if reasoning_memory_text is None and factor_memory_context:
            reasoning_memory_text = factor_memory_context
        if reasoning_memory_text:
            second_session_input += (
                "\n\nRelevant reusable reasoning memory is provided below."
                "\nExplicitly use the listed common factors and reasoning patterns while forming the forecast."
                "\nTreat the listed error patterns as mistakes you should actively avoid."
                "\nBefore finalizing the forecast, check whether your probability distribution repeats any listed calibration or reasoning error, and revise if needed."
                f"\n\n{reasoning_memory_text}"
            )
        second_session_input += "\n\nReturn the final JSON answer."

        result = None
        retry_prompt = second_session_input
        for attempt_idx in range(MAX_EMPTY_OUTPUT_RETRIES):
            result = await Runner.run(
                forecast_agent,
                retry_prompt,
                context=budget,
                hooks=BudgetHooks(),
                session=session,
                max_turns=max_turns,
            )
            final_output = result.final_output
            if isinstance(final_output, str) and final_output.strip():
                break
            if final_output not in (None, ""):
                break
            if attempt_idx + 1 < MAX_EMPTY_OUTPUT_RETRIES:
                print(
                    f"No final output returned for sample {task_idx}; retrying "
                    f"({attempt_idx + 2}/{MAX_EMPTY_OUTPUT_RETRIES})."
                )
                retry_prompt = (
                    "Your previous attempt returned no final answer."
                    "\nReturn the final JSON answer now."
                    "\nDo not continue open-ended reasoning."
                    "\nDo not search again unless it is strictly necessary."
                )
                if factor_decomposition is not None:
                    retry_prompt += (
                        "\n\nUse this factor decomposition:"
                        f"\n{json.dumps(factor_decomposition, ensure_ascii=True)}"
                    )
                if reasoning_memory_text:
                    retry_prompt += (
                        "\n\nRelevant reusable reasoning memory:"
                        f"\n{reasoning_memory_text}"
                    )
                retry_prompt += f"\n\nOriginal question:\n{question}"

        if result is None or result.final_output is None or (
            isinstance(result.final_output, str) and not result.final_output.strip()
        ):
            raise ValueError(
                f"No final output returned after {MAX_EMPTY_OUTPUT_RETRIES} attempts."
            )
        duration_seconds = time.perf_counter() - start_time

        print("searches used:", budget.search_calls_used)
        s2_usage = result.context_wrapper.usage
        _print_usage("session2", s2_usage.input_tokens, s2_usage.output_tokens, model_name)
        total_in = s1_usage.input_tokens + s2_usage.input_tokens
        total_out = s1_usage.output_tokens + s2_usage.output_tokens
        _print_usage("total", total_in, total_out, model_name)
        print(result.final_output)

        predictions = extract_probabilities_from_answer(
            result.final_output,
            sample_task["markets"],
        )
        processed_result: list[dict[str, Any]] = []
        for session_name, session_result in (
            ("session1", decomposition_result),
            ("session2", result),
        ):
            session_steps = inspect_agent_result(session_result, verbose=False)
            if session_name == "session1":
                session_steps = attach_decomposition_to_session_steps(
                    session_steps,
                    factor_decomposition,
                )
            for step in session_steps:
                step["session"] = session_name
            processed_result.extend(session_steps)
        serializable_result = {
            "status": "success",
            "idx": sample_task.get("idx"),
            "question": question,
            "outcomes": sample_task.get("market_outcome"),
            "market_options": sample_task["markets"],
            "predictions": predictions,
            "raw_answer": result.final_output,
            "model": model_name,
            "search_provider": search_provider,
            "max_turns": max_turns,
            "filter_year": filter_year,
            "filter_date_max": filter_date_max,
            "filter_days_before_close": filter_days_before_close,
            "use_close_date_filter": use_close_date_filter,
            "factor_decomposition": factor_decomposition,
            "used_external_memory": bool(external_memory_reasoning),
            "external_memory_metadata": external_memory_metadata or {},
            "retrieved_memory": {
                "factor_memory": (
                    {
                        "memory_type": "subcategory_factor_memory",
                        "metadata": external_memory_metadata or {},
                        "content_text": external_memory_factor or factor_memory_context,
                        "raw": external_memory_factor_raw or retrieved_factor_memory,
                    }
                    if (external_memory_factor or factor_memory_context)
                    else None
                ),
                "reasoning_memory": (
                    {
                        "memory_type": "subcategory_reasoning_memory",
                        "metadata": external_memory_metadata or {},
                        "content_text": external_memory_reasoning or factor_memory_context,
                        "raw": external_memory_reasoning_raw or retrieved_factor_memory,
                    }
                    if (external_memory_reasoning or factor_memory_context)
                    else None
                ),
            },
            "temperature": _model_temperature(model_name),
            "top_p": _model_top_p(model_name),
            "duration_seconds": duration_seconds,
            "timestamp": started_at,
            "used_experiences": False,
            "num_experiences": 0,
            "searches_used": budget.search_calls_used,
            "processed_result": processed_result,
            "token_usage": {
                "session1_input_tokens": s1_usage.input_tokens,
                "session1_output_tokens": s1_usage.output_tokens,
                "session2_input_tokens": s2_usage.input_tokens,
                "session2_output_tokens": s2_usage.output_tokens,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
            },
        }
        return {
            "result": result,
            "data": serializable_result,
        }
    except MaxTurnsExceeded as exc:
        duration_seconds = time.perf_counter() - start_time
        print("searches used:", budget.search_calls_used)
        print(f"Max turns exceeded for sample {task_idx}: {exc}")
        serializable_result = {
            "status": "max_turns_exceeded",
            "error": str(exc),
            "idx": sample_task.get("idx"),
            "question": question,
            "outcomes": sample_task.get("market_outcome"),
            "market_options": sample_task["markets"],
            "predictions": None,
            "raw_answer": None,
            "model": model_name,
            "search_provider": search_provider,
            "max_turns": max_turns,
            "filter_year": filter_year,
            "filter_date_max": filter_date_max,
            "filter_days_before_close": filter_days_before_close,
            "use_close_date_filter": use_close_date_filter,
            "factor_decomposition": factor_decomposition,
            "used_external_memory": bool(external_memory_reasoning),
            "external_memory_metadata": external_memory_metadata or {},
            "retrieved_memory": {
                "factor_memory": None,
                "reasoning_memory": None,
            },
            "temperature": _model_temperature(model_name),
            "top_p": _model_top_p(model_name),
            "duration_seconds": duration_seconds,
            "timestamp": started_at,
            "used_experiences": False,
            "num_experiences": 0,
            "searches_used": budget.search_calls_used,
            "processed_result": None,
            "token_usage": _partial_token_usage(s1_usage, s2_usage),
        }
        return {
            "result": None,
            "data": serializable_result,
        }
    except ValueError as exc:
        duration_seconds = time.perf_counter() - start_time
        print("searches used:", budget.search_calls_used)
        print(f"No final output for sample {task_idx}: {exc}")
        serializable_result = {
            "status": "no_output",
            "error": str(exc),
            "idx": sample_task.get("idx"),
            "question": question,
            "outcomes": sample_task.get("market_outcome"),
            "market_options": sample_task["markets"],
            "predictions": None,
            "raw_answer": None,
            "model": model_name,
            "search_provider": search_provider,
            "max_turns": max_turns,
            "filter_year": filter_year,
            "filter_date_max": filter_date_max,
            "filter_days_before_close": filter_days_before_close,
            "use_close_date_filter": use_close_date_filter,
            "factor_decomposition": factor_decomposition,
            "used_external_memory": bool(external_memory_reasoning),
            "external_memory_metadata": external_memory_metadata or {},
            "retrieved_memory": {
                "factor_memory": None,
                "reasoning_memory": None,
            },
            "temperature": _model_temperature(model_name),
            "top_p": _model_top_p(model_name),
            "duration_seconds": duration_seconds,
            "timestamp": started_at,
            "used_experiences": False,
            "num_experiences": 0,
            "searches_used": budget.search_calls_used,
            "processed_result": None,
            "token_usage": _partial_token_usage(s1_usage, s2_usage),
        }
        return {
            "result": None,
            "data": serializable_result,
        }


async def run_custom_question(
    question: str,
    markets: list[str],
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 20,
    search_provider: str = "serpapi",
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
) -> dict:
    """Run one prediction for a provided custom question."""
    custom_task = build_custom_task(question=question, markets=markets)
    output_dir = get_custom_output_dir(
        question,
        model_name=model_name,
        search_provider=search_provider,
        use_close_date_filter=use_close_date_filter,
    )
    result_path = output_dir / "result.json"
    clear_existing_output_files(result_path)

    prediction_result = await run_single_prediction(
        sample_task=custom_task,
        selected_week_csv="custom_input",
        task_idx=0,
        model_name=model_name,
        max_search_calls=max_search_calls,
        search_provider=search_provider,
        filter_year=filter_year,
        filter_days_before_close=filter_days_before_close,
        use_close_date_filter=use_close_date_filter,
    )

    result_payload = {
        **prediction_result["data"],
        "input_type": "custom_question",
    }
    result_path = save_custom_result(result_payload, output_dir)

    print()
    print(f"Saved custom question result to {result_path}")

    return {
        "output_dir": str(output_dir),
        "result_file": str(result_path),
        "data": result_payload,
    }


async def run_predictions(
    selected_week_idx: int | None = None,
    run_all_weeks: bool = False,
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 10,
    search_provider: str = "serpapi",
    use_existing_weekly_files: bool = USE_EXISTING_WEEKLY_FILES,
    weeks_limit: int = 10,
    first_k: int | None = None,
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
    max_concurrency: int = 5,
    resume_results_path: str | None = None,
    dataset: str = "prophet_arena",
    provider: str = "auto",
) -> list[dict]:
    """Run predictions for one selected week or for all available weeks."""
    client = _make_inference_client(provider)
    if client is not None:
        set_default_openai_client(client, use_for_tracing=False)

    if resume_results_path is not None and run_all_weeks:
        raise ValueError("--resume-results-path cannot be combined with --all-weeks.")

    if resume_results_path is not None:
        saved_results_payload = load_results_payload(resume_results_path)
        weekly_csvs = [str(saved_results_payload["selected_week_csv"])]
        week_indexes = [int(saved_results_payload.get("week_index", 0))]
    else:
        weekly_csvs = get_weekly_csvs(
            use_existing_weekly_files=use_existing_weekly_files,
            weeks_limit=weeks_limit,
            dataset=dataset,
        )

        if run_all_weeks:
            week_indexes = list(range(len(weekly_csvs)))
        else:
            idx = 0 if selected_week_idx is None else selected_week_idx
            # If idx matches a week number in any CSV name (e.g. 8 → week8), use that CSV.
            matched = next(
                (i for i, p in enumerate(weekly_csvs)
                 if re.search(rf"week0*{idx}[_.]", Path(p).name)),
                None,
            )
            week_indexes = [matched if matched is not None else idx]

    results = []
    for loop_idx, week_idx in enumerate(week_indexes):
        if resume_results_path is not None:
            selected_week_csv = weekly_csvs[loop_idx]
        else:
            selected_week_csv = weekly_csvs[week_idx]
        print()
        print(f"=== Running week {week_idx}: {selected_week_csv} ===")

        weekly_tasks = load_weekly_tasks(selected_week_csv, first_k=first_k)
        if not weekly_tasks:
            raise ValueError(
                f"No tasks are available in {selected_week_csv}."
            )

        week_output_dir = get_week_output_dir(
            selected_week_csv,
            model_name=model_name,
            search_provider=search_provider,
            use_close_date_filter=use_close_date_filter,
            dataset=dataset,
        )
        results_by_idx: dict[int, dict[str, Any]] = {}
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        if resume_results_path is not None:
            results_path = Path(resume_results_path)
            if results_path.parent != week_output_dir:
                week_output_dir = results_path.parent
        else:
            results_path = resolve_results_path(week_output_dir, use_close_date_filter)
            clear_existing_output_files(results_path)

        start_task_idx = 0
        pending_rerun_indices: list[int] = []
        if resume_results_path is not None:
            existing_results_payload = load_results_payload(results_path)
            saved_samples = copy.deepcopy(existing_results_payload.get("samples", []))
            start_task_idx = len(saved_samples)
            if start_task_idx > len(weekly_tasks):
                raise ValueError(
                    f"Saved results contain {start_task_idx} samples, but only {len(weekly_tasks)} tasks are available."
                )

            pending_rerun_indices = [
                idx for idx, saved_sample in enumerate(saved_samples) if _sample_needs_rerun(saved_sample)
            ]
            for idx, saved_sample in enumerate(saved_samples):
                if idx not in pending_rerun_indices:
                    results_by_idx[idx] = saved_sample

            saved_max_search_calls = existing_results_payload.get("max_search_calls")
            if (
                saved_max_search_calls is not None
                and int(saved_max_search_calls) != int(max_search_calls)
            ):
                print(
                    f"Saved results use max_search_calls={saved_max_search_calls}, "
                    f"but the current run uses {max_search_calls}. "
                    "Keeping successful samples and re-running saved error samples only."
                )

            print(
                f"Resuming from {results_path} with {start_task_idx} saved task(s), "
                f"{len(pending_rerun_indices)} needing rerun."
            )

        async def _run_one(task_idx: int, sample_task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                max_retries = 20
                for attempt in range(max_retries):
                    try:
                        prediction_result = await run_single_prediction(
                            sample_task=sample_task,
                            selected_week_csv=selected_week_csv,
                            task_idx=task_idx,
                            model_name=model_name,
                            max_search_calls=max_search_calls,
                            search_provider=search_provider,
                            filter_year=filter_year,
                            filter_days_before_close=filter_days_before_close,
                            use_close_date_filter=use_close_date_filter,
                        )
                        break
                    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                        if attempt == max_retries - 1:
                            raise
                        wait = min(2 ** attempt * 10, 180)
                        print(f"  [task {task_idx}] transient error, retrying in {wait}s ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait)
            return task_idx, prediction_result["data"]

        task_indices_to_run = pending_rerun_indices + list(range(start_task_idx, len(weekly_tasks)))
        pending_tasks = [
            asyncio.create_task(_run_one(task_idx, sample_task))
            for task_idx, sample_task in (
                (task_idx, weekly_tasks[task_idx]) for task_idx in task_indices_to_run
            )
        ]

        for completed_task in asyncio.as_completed(pending_tasks):
            task_idx, sample_result = await completed_task
            results_by_idx[task_idx] = sample_result
            sample_results = [results_by_idx[idx] for idx in sorted(results_by_idx)]
            week_results_payload = build_week_results_payload(
                selected_week_csv=selected_week_csv,
                week_idx=week_idx,
                model_name=model_name,
                max_search_calls=max_search_calls,
                search_provider=search_provider,
                filter_year=filter_year,
                filter_days_before_close=filter_days_before_close,
                use_close_date_filter=use_close_date_filter,
                sample_results=sample_results,
            )
            results_path = save_week_results(
                week_results_payload,
                week_output_dir,
                use_close_date_filter=use_close_date_filter,
            )

        sample_results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

        results.append(
            {
                "selected_week_csv": selected_week_csv,
                "week_index": week_idx,
                "week_output_dir": str(week_output_dir),
                "num_samples": len(sample_results),
                "search_provider": search_provider,
                "filter_year": filter_year,
                "filter_days_before_close": filter_days_before_close,
                "use_close_date_filter": use_close_date_filter,
                "samples": sample_results,
            }
        )

    print()
    print("=== Summary ===")
    for idx, result in zip(week_indexes, results):
        print(
            f"week index {idx}: {Path(result['selected_week_csv']).name} "
            f"-> {result['num_samples']} samples saved"
        )

    return results


__all__ = [
    "MAX_EMPTY_OUTPUT_RETRIES",
    "run_custom_question",
    "run_predictions",
    "run_predictions_with_memory",
    "run_single_prediction",
]
