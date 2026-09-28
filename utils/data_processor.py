import copy
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


def extract_json_object_from_text(answer: str, context: str = "response") -> dict[str, Any]:
    """Extract a JSON object from model text output."""
    if not isinstance(answer, str):
        raise TypeError(f"Expected text output for {context}, got {type(answer).__name__}.")

    if "{" not in answer or "}" not in answer:
        raise ValueError(f"No JSON object found in {context}.")

    start_idx = answer.find("{")
    end_idx = answer.rfind("}") + 1
    json_str = answer[start_idx:end_idx].strip()

    if json_str.startswith("```"):
        json_str = re.sub(r"^```(?:json)?\s*", "", json_str, count=1)
        json_str = re.sub(r"\s*```$", "", json_str, count=1)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = json.loads(json_str, strict=False)

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected JSON object in {context}, got {type(parsed).__name__}.")
    return parsed


def extract_probabilities_from_answer(answer: str, markets: list) -> dict:
    """Extract probability predictions from the model's answer."""
    try:
        if "{" in answer and "}" in answer:
            parsed = extract_json_object_from_text(
                answer,
                context="forecast answer",
            )

            if "probabilities" in parsed:
                probs = parsed["probabilities"]
            elif all(market in parsed for market in markets):
                probs = parsed
            else:
                probs = {}
                for market in markets:
                    if market in parsed:
                        probs[market] = float(parsed[market])

            result = {}
            for market in markets:
                if market in probs and probs[market] is not None:
                    result[market] = float(probs[market])
                else:
                    result[market] = 1.0 / len(markets)

            return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse probabilities from answer: {e}")

    return {market: 1.0 / len(markets) for market in markets}


def attach_decomposition_to_session_steps(
    session_steps: list[dict[str, Any]],
    decomposition: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach the normalized decomposition payload to session-1 message steps."""
    if not session_steps or not decomposition:
        return session_steps

    enriched_steps = copy.deepcopy(session_steps)
    for step in enriched_steps:
        if step.get("item_type") != "message":
            continue
        step["parsed_json"] = copy.deepcopy(decomposition)
        step["decomposed_factors"] = copy.deepcopy(decomposition.get("factors", []))
        step["message_text"] = json.dumps(decomposition, ensure_ascii=True)
    return enriched_steps


def parse_outcomes_arg(outcomes_arg: str | None) -> list[str]:
    """Parse custom outcomes from JSON or comma-separated text."""
    if not outcomes_arg:
        return ["Yes", "No"]

    try:
        parsed = json.loads(outcomes_arg)
        if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed = [item.strip() for item in outcomes_arg.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Custom outcomes must be a non-empty JSON list or comma-separated string.")
    return parsed


def build_custom_task(question: str, markets: list[str]) -> dict[str, Any]:
    """Create a task payload for a custom input question."""
    return {
        "idx": 0,
        "event_ticker": "custom_question",
        "title": question,
        "category": "custom",
        "markets": markets,
    }

def build_week_results_payload(
    selected_week_csv: str,
    week_idx: int,
    model_name: str,
    max_search_calls: int,
    search_provider: str,
    filter_year: int | None,
    filter_days_before_close: int,
    use_close_date_filter: bool,
    sample_results: list[dict],
) -> dict:
    """Build the combined results payload for one week."""
    max_turns = max_search_calls
    return {
        "selected_week_csv": selected_week_csv,
        "week_index": week_idx,
        "model": model_name,
        "max_search_calls": max_search_calls,
        "max_turns": max_turns,
        "search_provider": search_provider,
        "filter_year": filter_year,
        "filter_days_before_close": filter_days_before_close,
        "use_close_date_filter": use_close_date_filter,
        "num_samples": len(sample_results),
        "samples": sample_results,
    }
