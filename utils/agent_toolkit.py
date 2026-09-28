import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from agents import Agent, ModelSettings, RunContextWrapper, RunHooks, Runner
from agents.tool import (
    SerpAPISearchTool,
    SerperSearchTool,
    TavilySearchTool,
    _serpapi_google_search_impl,
    _serper_google_search_impl,
    _tavily_google_search_impl,
    function_tool,
)
from openai.types.shared import Reasoning
from utils.data_processor import (
    attach_decomposition_to_session_steps,
    build_custom_task,
    build_week_results_payload,
    extract_json_object_from_text,
    extract_probabilities_from_answer,
    parse_outcomes_arg,
)
from utils.factor_memory import (
    DEFAULT_FACTOR_MEMORY_CONFIG,
    FACTOR_MEMORY_SELF_CORRECT_PROMPT_PATH,
    FactorMemoryConfig,
    build_factor_memory_summary,
    load_factor_memory,
    load_factor_memory_self_correct_prompt,
    save_factor_memory,
    save_factor_memory_history,
)
from utils.file_processor import (
    PROJECT_ROOT,
    RESULTS_DIR,
    USE_EXISTING_WEEKLY_FILES,
    clear_existing_output_files,
    get_custom_output_dir,
    get_results_filename,
    get_week_output_dir,
    get_weekly_csvs,
    load_results_payload,
    load_weekly_tasks,
    resolve_existing_results_input_path,
    resolve_results_path,
    save_custom_result,
    save_week_results,
)


load_dotenv()

DEFAULT_MODEL_TEMPERATURE = 0.7
DEFAULT_MODEL_TOP_P = 0.7


def _model_temperature(model_name: str) -> float | None:
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return None
    return DEFAULT_MODEL_TEMPERATURE


def _model_top_p(model_name: str) -> float | None:
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return None
    return DEFAULT_MODEL_TOP_P


class FactorDecompositionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_name: str
    rationale: str
    typical_effect_on_output: str = ""
    potential_common_error_pattern: str = ""


class FactorDecompositionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factors: list[FactorDecompositionItem] = Field(default_factory=list)


class FactorMemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_name: str
    notes: list[str] = Field(default_factory=list)
    update_cues: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    effect: str = ""
    failure_pattern: str = ""
    utility: float = 1.0


class FactorMemorySelfCorrectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used_factors: list[FactorMemoryItem] = Field(default_factory=list)
    useful_factors: list[FactorMemoryItem] = Field(default_factory=list)
    overweighted_factors: list[FactorMemoryItem] = Field(default_factory=list)
    missed_factors: list[FactorMemoryItem] = Field(default_factory=list)
    most_influential_factor: str = ""
    most_misleading_factor: str = ""
    correction_note: str = ""


@dataclass
class ForecastBudget:
    max_search_calls: int = 3
    search_calls_used: int = 0
    task_idx: int | None = None
    last_search_query: str = ""


def search_tool_enabled(ctx: RunContextWrapper[ForecastBudget], agent) -> bool:
    return ctx.context.search_calls_used < ctx.context.max_search_calls


def build_search_tool(
    search_provider: str = "serpapi",
    filter_year: int | None = None,
    filter_date_max: str | None = None,
):
    """Build a configured web search tool for the selected provider."""
    if search_provider not in {"serpapi", "serper", "tavily"}:
        raise ValueError(f"Unsupported search provider: {search_provider}")

    provider_configs = {
        "serpapi": {
            "base_tool": SerpAPISearchTool,
            "search_impl": _serpapi_google_search_impl,
            "tool_name": "serpapi_google_search",
            "provider_label": "SerpAPI",
        },
        "serper": {
            "base_tool": SerperSearchTool,
            "search_impl": _serper_google_search_impl,
            "tool_name": "serper_google_search",
            "provider_label": "Serper.dev",
        },
        "tavily": {
            "base_tool": TavilySearchTool,
            "search_impl": _tavily_google_search_impl,
            "tool_name": "tavily_google_search",
            "provider_label": "Tavily",
        },
    }
    provider_config = provider_configs[search_provider]
    base_tool = provider_config["base_tool"]
    search_impl = provider_config["search_impl"]
    tool_name = provider_config["tool_name"]
    provider_label = provider_config["provider_label"]

    if filter_date_max is not None:
        description = (
            f"Performs a Google web search via {provider_label} and returns the top results as text. "
            f"Results are restricted to sources on or before {filter_date_max}."
        )
    elif filter_year is not None:
        description = (
            f"Performs a Google web search via {provider_label} and returns the top results as text. "
            f"Results are restricted to the year {filter_year}."
        )
    else:
        description = base_tool.description

    def tracked_search(ctx: RunContextWrapper[ForecastBudget], query: str, max_results: int = 5) -> str:
        ctx.context.last_search_query = query
        return search_impl(
            query=query,
            max_results=max_results,
            filter_year=filter_year,
            filter_date_max=filter_date_max,
        )

    tool = function_tool(
        tracked_search,
        name_override=tool_name,
        description_override=description,
    )
    tool.is_enabled = search_tool_enabled
    return tool


_SEARCH_TOOL_NAMES = {
    "serpapi_google_search",
    "serper_google_search",
    "tavily_google_search",
}


class BudgetHooks(RunHooks[ForecastBudget]):
    async def on_tool_start(self, context, agent, tool) -> None:
        if tool.name in _SEARCH_TOOL_NAMES:
            context.context.search_calls_used += 1
            task_label = (
                f"task {context.context.task_idx} "
                if context.context.task_idx is not None
                else ""
            )
            print(
                f"{task_label}search {context.context.search_calls_used}/"
                f"{context.context.max_search_calls}",
                flush=True,
            )

    async def on_tool_end(self, context, agent, tool, result) -> None:
        if tool.name not in _SEARCH_TOOL_NAMES:
            return
        task_label = (
            f"task {context.context.task_idx} "
            if context.context.task_idx is not None
            else ""
        )
        query = context.context.last_search_query
        try:
            result_text = str(result.output if hasattr(result, "output") else result)
            result_preview = result_text[:300].replace("\n", " ")
        except Exception:
            result_preview = "(unable to read result)"
        print(
            f"{task_label}search result | query={query!r} | preview={result_preview!r}",
            flush=True,
        )


async def trajectory_self_corrector_with_backbone(
    prediction_result: Any,
    model_name: str = "gpt-5-mini",
) -> dict[str, Any]:
    """Use the forecasting backbone model to self-correct a trajectory into factor hindsight."""
    if not isinstance(prediction_result, dict):
        raise TypeError("prediction_result must be a dict containing trajectory and task context.")

    processed_result = prediction_result.get("processed_result")
    if isinstance(processed_result, dict) and processed_result.get("used_factors") is not None:
        return processed_result

    prompt_template = load_factor_memory_self_correct_prompt()
    question = prediction_result.get("question", "")
    title = prediction_result.get("title", "")
    body = prediction_result.get("body", "")
    resolution_date = prediction_result.get("resolution_date", "")
    created_date = prediction_result.get("created_date", "")
    ground_truth = prediction_result.get("ground_truth")
    outcome = prediction_result.get("outcome")
    full_response = prediction_result.get("full_response", "")
    trajectory_text = json.dumps(prediction_result.get("processed_result"), ensure_ascii=False, indent=2)

    prompt = prompt_template.format(
        title=title,
        question=question,
        body=body,
        resolution_date=resolution_date,
        created_date=created_date,
        ground_truth=json.dumps(ground_truth, ensure_ascii=False, indent=2) if ground_truth is not None else "null",
        outcome=json.dumps(outcome, ensure_ascii=False, indent=2) if outcome is not None else "null",
        full_response=str(full_response),
        trajectory=trajectory_text,
    )

    _supports_reasoning = model_name.startswith(("gpt-5", "o1", "o3", "o4"))
    audit_agent = Agent(
        name="factor_memory_self_corrector",
        instructions="Return only valid JSON that matches the requested schema.",
        model=model_name,
        model_settings=ModelSettings(
            temperature=_model_temperature(model_name),
            top_p=_model_top_p(model_name),
            reasoning=Reasoning(effort="medium") if _supports_reasoning else None,
        ),
        output_type=FactorMemorySelfCorrectionOutput,
    )
    audit_result = await Runner.run(audit_agent, prompt)
    return audit_result.final_output.model_dump()


__all__ = [
    "BudgetHooks",
    "DEFAULT_FACTOR_MEMORY_CONFIG",
    "FACTOR_MEMORY_SELF_CORRECT_PROMPT_PATH",
    "FactorDecompositionItem",
    "FactorDecompositionOutput",
    "FactorMemoryConfig",
    "FactorMemoryItem",
    "FactorMemorySelfCorrectionOutput",
    "ForecastBudget",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "USE_EXISTING_WEEKLY_FILES",
    "attach_decomposition_to_session_steps",
    "build_custom_task",
    "build_factor_memory_summary",
    "build_search_tool",
    "build_week_results_payload",
    "clear_existing_output_files",
    "extract_json_object_from_text",
    "extract_probabilities_from_answer",
    "get_custom_output_dir",
    "get_results_filename",
    "get_week_output_dir",
    "get_weekly_csvs",
    "load_factor_memory",
    "load_factor_memory_self_correct_prompt",
    "load_results_payload",
    "load_weekly_tasks",
    "parse_outcomes_arg",
    "resolve_existing_results_input_path",
    "resolve_results_path",
    "save_custom_result",
    "save_factor_memory",
    "save_factor_memory_history",
    "save_week_results",
    "search_tool_enabled",
    "trajectory_self_corrector_with_backbone",
]
