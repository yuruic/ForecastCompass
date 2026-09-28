import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompt"
FACTOR_MEMORY_SELF_CORRECT_PROMPT_PATH = PROMPT_DIR / "factor_memory_self_correct.txt"

DEFAULT_FACTOR_MEMORY_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class FactorMemoryConfig:
    usage_decay: float = 0.95
    recency_decay: float = 0.90
    utility_momentum: float = 0.80
    persistence_momentum: float = 0.90
    redundancy_penalty: float = 0.20
    active_threshold: float = 2.50
    weak_threshold: float = 1.25
    delete_threshold: float = 0.50
    delete_after_consecutive: int = 3


DEFAULT_FACTOR_MEMORY_CONFIG = FactorMemoryConfig()


# Type alias for a trajectory self-correction function
TrajectorySelfCorrector = Callable[[Any], Awaitable[dict[str, Any]]]


def load_factor_memory_self_correct_prompt() -> str:
    """Load the factor-memory self-correction prompt from disk."""
    return FACTOR_MEMORY_SELF_CORRECT_PROMPT_PATH.read_text(encoding="utf-8")


def save_factor_memory(memory_payload: dict, output_dir: str | Path) -> Path:
    """Save factor memory as JSON."""
    output_path = Path(output_dir) / "factor_memory.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(memory_payload, f, indent=2, ensure_ascii=True)
    return output_path


def save_factor_memory_history(history_payload: dict, output_dir: str | Path) -> Path:
    """Save per-task factor memory snapshots as JSON."""
    output_path = Path(output_dir) / "factor_memory_history.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2, ensure_ascii=True)
    return output_path


def load_factor_memory(memory_path: str | Path) -> dict[str, Any]:
    """Load factor memory JSON from disk."""
    memory_path = Path(memory_path)
    with memory_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_factor_memory_summary(memory_payload: dict) -> dict[str, Any]:
    """Build a compact summary for logging and downstream inspection."""
    factor_slots = memory_payload.get("factor_slots", {})
    active_factors = []
    weak_factors = []
    for factor_key, slot in factor_slots.items():
        status = slot.get("statistics", {}).get("status")
        if status == "active":
            active_factors.append(factor_key)
        elif status == "weak":
            weak_factors.append(factor_key)

    return {
        "memory_type": memory_payload.get("memory_type"),
        "num_processed_results": memory_payload.get("num_processed_results", 0),
        "num_factor_slots": len(factor_slots),
        "active_factors": active_factors,
        "weak_factors": weak_factors,
    }


def normalize_factor_name(name: str) -> str:
    """Normalize a factor name into a stable key."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "unknown_factor"


COMMON_FACTOR_NAME_ALIASES = {
    "team_success": "team_advancement",
    "team_progression": "team_advancement",
    "advancement": "team_advancement",
    "player_impact": "player_centrality",
    "attribution": "player_centrality",
    "centrality": "player_centrality",
    "narrative": "narrative_salience",
    "visibility": "tournament_visibility",
    "market_demand": "demand_momentum",
    "demand": "demand_momentum",
    "regulation": "regulatory_support",
    "regulatory": "regulatory_support",
    "execution": "execution_feasibility",
    "feasibility": "execution_feasibility",
}


def canonicalize_factor_name(name: str) -> str:
    """Map factor names to a canonical memory key."""
    normalized = normalize_factor_name(name)
    return COMMON_FACTOR_NAME_ALIASES.get(normalized, normalized)


DEFAULT_FACTOR_SEMANTICS = {
    "team_advancement": {
        "factor_name": "Team Advancement",
        "meaning": "Measures whether the predicted outcome depends on deep progression, broad team success, or strong structural positioning.",
        "typical_role": "Acts as a structural support or gating factor for competitive outcome forecasts.",
        "typical_update_cues": [
            "stronger progression outlook",
            "reduced elimination risk",
            "evidence of durable team-level success",
        ],
        "typical_effect": "Usually increases the probability of candidate-level success when structural progression remains favorable.",
        "common_interactions": [
            "amplifies tournament visibility",
            "amplifies player centrality",
        ],
        "common_failure_pattern": "Do not overestimate individual upside when broader team success is fragile.",
    },
    "player_centrality": {
        "factor_name": "Player Centrality",
        "meaning": "Measures whether success can be clearly attributed to a focal player, actor, or key decision-maker.",
        "typical_role": "Acts as an attribution factor in candidate-level or actor-level forecasting.",
        "typical_update_cues": [
            "clear carry dependence",
            "repeated decisive involvement",
            "evidence that outcomes are concentrated around one actor",
        ],
        "typical_effect": "Usually increases forecast confidence when attribution becomes easier and more stable.",
        "common_interactions": [
            "strengthened by team advancement",
            "can be confused with narrative salience",
        ],
        "common_failure_pattern": "Do not confuse general skill or attention with clearly attributable central impact.",
    },
    "tournament_visibility": {
        "factor_name": "Tournament Visibility",
        "meaning": "Measures whether an outcome driver is becoming highly salient in decisive or high-pressure settings.",
        "typical_role": "Acts as a late-separation factor in competitive forecasting.",
        "typical_update_cues": [
            "high-salience performance",
            "repeated highlight-worthy impact",
            "strong visibility in decisive settings",
        ],
        "typical_effect": "Can sharply boost a forecast when structural conditions are already favorable.",
        "common_interactions": [
            "amplified by team advancement",
            "amplified by narrative salience",
        ],
        "common_failure_pattern": "Do not overweight a single visibility spike without broader support.",
    },
    "narrative_salience": {
        "factor_name": "Narrative Salience",
        "meaning": "Measures whether a compelling storyline is increasing attention, memorability, or subjective separation.",
        "typical_role": "Acts as a secondary separation factor when primary drivers are close.",
        "typical_update_cues": [
            "storyline consolidation",
            "strong public framing",
            "alignment between performance and public perception",
        ],
        "typical_effect": "Usually has limited standalone value but can separate close candidates or outcomes.",
        "common_interactions": [
            "amplified by tournament visibility",
            "should be downweighted if structural support is weak",
        ],
        "common_failure_pattern": "Do not treat narrative as primary before stronger structural factors are established.",
    },
    "regulatory_support": {
        "factor_name": "Regulatory Support",
        "meaning": "Measures whether the institutional environment is moving in a direction supportive of the forecasted outcome.",
        "typical_role": "Acts as a slow-moving structural driver in policy and approval forecasting.",
        "typical_update_cues": [
            "official statements",
            "rulemaking progress",
            "alignment from decision-makers",
        ],
        "typical_effect": "Usually shifts the baseline forecast direction when signals are formal and sustained.",
        "common_interactions": [
            "gated by execution feasibility",
            "can be offset by stakeholder opposition",
        ],
        "common_failure_pattern": "Do not equate vague commentary with formal regulatory movement.",
    },
    "execution_feasibility": {
        "factor_name": "Execution Feasibility",
        "meaning": "Measures whether the forecasted outcome can realistically be implemented under the relevant timeline and resource constraints.",
        "typical_role": "Acts as a bottleneck factor that converts support into real deliverability.",
        "typical_update_cues": [
            "milestone completion",
            "operational readiness",
            "timeline slippage or acceleration",
        ],
        "typical_effect": "Can strongly lower forecast probability when feasibility is weak, even if support is positive.",
        "common_interactions": [
            "gates regulatory support",
            "gates demand momentum",
        ],
        "common_failure_pattern": "Do not confuse positive intent with near-term deliverability.",
    },
    "demand_momentum": {
        "factor_name": "Demand Momentum",
        "meaning": "Measures whether demand or adoption signals are moving in a sustained direction that supports future growth.",
        "typical_role": "Acts as a medium-moving directional growth factor.",
        "typical_update_cues": [
            "repeat adoption signals",
            "engagement growth",
            "cross-source confirmation of demand strength",
        ],
        "typical_effect": "Usually raises outcome probability when growth appears durable rather than episodic.",
        "common_interactions": [
            "strengthened by infrastructure readiness",
            "offset by supply or execution constraints",
        ],
        "common_failure_pattern": "Do not treat short-term hype as sustained demand.",
    },
}


def _make_default_factor_slot(factor_key: str) -> dict[str, Any]:
    semantic_defaults = DEFAULT_FACTOR_SEMANTICS.get(
        factor_key,
        {
            "factor_name": factor_key.replace("_", " ").title(),
            "meaning": "Reusable latent factor for forecasting tasks.",
            "typical_role": "Acts as a forecasting driver or constraint depending on context.",
            "typical_update_cues": [],
            "typical_effect": "Can shift event-level belief when supported by repeated evidence.",
            "common_interactions": [],
            "common_failure_pattern": "Avoid over-weighting this factor without repeated confirmation.",
        },
    )
    return {
        "factor_key": factor_key,
        "factor_name": semantic_defaults["factor_name"],
        "meaning": semantic_defaults["meaning"],
        "typical_role": semantic_defaults["typical_role"],
        "typical_update_cues": list(semantic_defaults["typical_update_cues"]),
        "typical_effect": semantic_defaults["typical_effect"],
        "common_interactions": list(semantic_defaults["common_interactions"]),
        "common_failure_pattern": semantic_defaults["common_failure_pattern"],
        "statistics": {
            "usage_score": 0.0,
            "recency_score": 0.0,
            "persistence_score": 0.0,
            "utility_score": 0.0,
            "redundancy_score": 0.0,
            "retention_score": 0.0,
            "last_used_step": None,
            "consecutive_low_retention": 0,
            "status": "active",
        },
    }


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_text_list(values: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _extract_factor_items(value: Any, default_label: str = "unknown") -> list[dict[str, Any]]:
    items = _safe_list(value)
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            factor_name = item.get("factor") or item.get("name") or item.get("factor_name") or default_label
            normalized.append(
                {
                    "factor_name": str(factor_name),
                    "notes": _coerce_text_list(
                        _safe_list(item.get("notes"))
                        + _safe_list(item.get("reason"))
                        + _safe_list(item.get("rationale"))
                        + _safe_list(item.get("summary"))
                    ),
                    "update_cues": _coerce_text_list(
                        _safe_list(item.get("update_cues"))
                        + _safe_list(item.get("signals"))
                        + _safe_list(item.get("evidence_types"))
                    ),
                    "interactions": _coerce_text_list(
                        _safe_list(item.get("interactions"))
                        + _safe_list(item.get("combined_effects"))
                    ),
                    "effect": str(item.get("effect", "")).strip(),
                    "failure_pattern": str(item.get("failure_pattern", "")).strip(),
                    "utility": float(item.get("utility", item.get("score", 1.0))),
                }
            )
        else:
            normalized.append(
                {
                    "factor_name": str(item),
                    "notes": [],
                    "update_cues": [],
                    "interactions": [],
                    "effect": "",
                    "failure_pattern": "",
                    "utility": 1.0,
                }
            )
    return normalized


def summarize_factor_hindsight_from_self_correct(self_correct_result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a model-produced self-correction result into factor-memory update signals."""
    self_correct_result = self_correct_result or {}

    used_candidates = _extract_factor_items(
        self_correct_result.get("used_factors")
        or self_correct_result.get("selected_factors")
        or self_correct_result.get("factors")
        or self_correct_result.get("factor_candidates")
    )
    useful_candidates = _extract_factor_items(
        self_correct_result.get("useful_factors") or self_correct_result.get("helpful_factors")
    )
    overweighted_candidates = _extract_factor_items(
        self_correct_result.get("overweighted_factors") or self_correct_result.get("misleading_factors")
    )
    missed_candidates = _extract_factor_items(
        self_correct_result.get("missed_factor_candidates") or self_correct_result.get("missed_factors")
    )

    used_keys = {canonicalize_factor_name(item["factor_name"]) for item in used_candidates}
    useful_keys = {canonicalize_factor_name(item["factor_name"]) for item in useful_candidates}
    overweighted_keys = {canonicalize_factor_name(item["factor_name"]) for item in overweighted_candidates}
    missed_keys = {canonicalize_factor_name(item["factor_name"]) for item in missed_candidates}

    factor_updates: dict[str, dict[str, Any]] = {}
    for source_name, items in {
        "used": used_candidates,
        "useful": useful_candidates,
        "overweighted": overweighted_candidates,
        "missed": missed_candidates,
    }.items():
        for item in items:
            factor_key = canonicalize_factor_name(item["factor_name"])
            update = factor_updates.setdefault(
                factor_key,
                {
                    "factor_key": factor_key,
                    "factor_name": item["factor_name"],
                    "used": False,
                    "useful": False,
                    "overweighted": False,
                    "missed": False,
                    "notes": [],
                    "update_cues": [],
                    "interactions": [],
                    "effects": [],
                    "failure_patterns": [],
                    "utility_values": [],
                },
            )
            update[source_name if source_name != "used" else "used"] = True
            update["notes"].extend(item.get("notes", []))
            update["update_cues"].extend(item.get("update_cues", []))
            update["interactions"].extend(item.get("interactions", []))
            if item.get("effect"):
                update["effects"].append(item["effect"])
            if item.get("failure_pattern"):
                update["failure_patterns"].append(item["failure_pattern"])
            update["utility_values"].append(float(item.get("utility", 1.0)))

    most_influential = self_correct_result.get("most_influential_factor")
    if most_influential:
        factor_key = canonicalize_factor_name(str(most_influential))
        update = factor_updates.setdefault(
            factor_key,
            {
                "factor_key": factor_key,
                "factor_name": str(most_influential),
                "used": True,
                "useful": True,
                "overweighted": False,
                "missed": False,
                "notes": [],
                "update_cues": [],
                "interactions": [],
                "effects": [],
                "failure_patterns": [],
                "utility_values": [],
            },
        )
        update["used"] = True
        update["useful"] = True
        update["utility_values"].append(1.0)

    most_misleading = self_correct_result.get("most_misleading_factor")
    if most_misleading:
        factor_key = canonicalize_factor_name(str(most_misleading))
        update = factor_updates.setdefault(
            factor_key,
            {
                "factor_key": factor_key,
                "factor_name": str(most_misleading),
                "used": True,
                "useful": False,
                "overweighted": True,
                "missed": False,
                "notes": [],
                "update_cues": [],
                "interactions": [],
                "effects": [],
                "failure_patterns": [],
                "utility_values": [],
            },
        )
        update["used"] = True
        update["overweighted"] = True
        update["utility_values"].append(0.0)

    correction_note = str(self_correct_result.get("correction_note", "")).strip()
    if correction_note:
        for factor_key in useful_keys | missed_keys | overweighted_keys:
            factor_updates.setdefault(
                factor_key,
                {
                    "factor_key": factor_key,
                    "factor_name": factor_key.replace("_", " ").title(),
                    "used": factor_key in used_keys,
                    "useful": factor_key in useful_keys,
                    "overweighted": factor_key in overweighted_keys,
                    "missed": factor_key in missed_keys,
                    "notes": [],
                    "update_cues": [],
                    "interactions": [],
                    "effects": [],
                    "failure_patterns": [],
                    "utility_values": [],
                },
            )["notes"].append(correction_note)

    return {
        "used_factor_keys": used_keys,
        "useful_factor_keys": useful_keys,
        "overweighted_factor_keys": overweighted_keys,
        "missed_factor_keys": missed_keys,
        "factor_updates": factor_updates,
        "correction_note": correction_note,
    }
async def self_correct_trajectory(
    prediction_result: Any,
    trajectory_self_corrector: TrajectorySelfCorrector,
) -> dict[str, Any]:
    """Use the agent backbone model to convert a raw trajectory into hindsight factor supervision."""
    if trajectory_self_corrector is None:
        raise ValueError(
            "trajectory_self_corrector is required because processed_result contains the raw agent trajectory."
        )

    self_correct_result = await trajectory_self_corrector(prediction_result)
    if not isinstance(self_correct_result, dict):
        raise TypeError(
            "trajectory_self_corrector must return a dict with factor self-correction fields."
        )
    return self_correct_result


def determine_factor_update_order(hindsight_summary: dict[str, Any]) -> list[str]:
    priority_scores: dict[str, float] = {}
    for factor_key, update in hindsight_summary.get("factor_updates", {}).items():
        score = 0.0
        if update.get("useful"):
            score += 4.0
        if update.get("used"):
            score += 2.0
        if update.get("missed"):
            score += 3.0
        if update.get("overweighted"):
            score += 1.0
        score += 0.1 * len(update.get("update_cues", []))
        score += 0.1 * len(update.get("interactions", []))
        priority_scores[factor_key] = score

    return sorted(priority_scores, key=lambda factor_key: (-priority_scores[factor_key], factor_key))


def _merge_unique(existing: list[str], new_items: Iterable[str], max_items: int = 8) -> list[str]:
    merged = list(existing)
    seen = {item.strip().lower() for item in merged if item.strip()}
    for item in new_items:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(text)
        if len(merged) >= max_items:
            break
    return merged[:max_items]


def update_factor_semantics(slot: dict[str, Any], factor_update: dict[str, Any]) -> None:
    if factor_update.get("factor_name"):
        slot["factor_name"] = factor_update["factor_name"]

    notes = _coerce_text_list(factor_update.get("notes", []))
    if notes:
        slot["meaning"] = notes[0]

    slot["typical_update_cues"] = _merge_unique(
        slot.get("typical_update_cues", []),
        factor_update.get("update_cues", []),
    )
    slot["common_interactions"] = _merge_unique(
        slot.get("common_interactions", []),
        factor_update.get("interactions", []),
    )

    effects = _coerce_text_list(factor_update.get("effects", []))
    if effects:
        slot["typical_effect"] = effects[0]

    failure_patterns = _coerce_text_list(factor_update.get("failure_patterns", []))
    if failure_patterns:
        slot["common_failure_pattern"] = failure_patterns[0]
    elif factor_update.get("overweighted") and notes:
        slot["common_failure_pattern"] = notes[0]


def update_factor_statistics(
    slot: dict[str, Any],
    factor_update: dict[str, Any] | None,
    step_idx: int,
    config: FactorMemoryConfig,
) -> None:
    stats = slot.setdefault("statistics", {})
    stats.setdefault("usage_score", 0.0)
    stats.setdefault("recency_score", 0.0)
    stats.setdefault("persistence_score", 0.0)
    stats.setdefault("utility_score", 0.0)
    stats.setdefault("redundancy_score", 0.0)
    stats.setdefault("retention_score", 0.0)
    stats.setdefault("last_used_step", None)
    stats.setdefault("consecutive_low_retention", 0)
    stats.setdefault("status", "active")

    stats["usage_score"] *= config.usage_decay
    stats["recency_score"] *= config.recency_decay

    was_used = bool(factor_update and factor_update.get("used"))
    was_useful = bool(factor_update and factor_update.get("useful"))
    was_overweighted = bool(factor_update and factor_update.get("overweighted"))
    was_missed = bool(factor_update and factor_update.get("missed"))

    if was_used:
        stats["usage_score"] += 1.0
        stats["recency_score"] = 1.0
        stats["last_used_step"] = step_idx

    utility_signal = 0.0
    if factor_update:
        utility_values = factor_update.get("utility_values") or []
        if utility_values:
            utility_signal = sum(utility_values) / len(utility_values)
        elif was_useful:
            utility_signal = 1.0
        elif was_overweighted:
            utility_signal = 0.0
        elif was_used:
            utility_signal = 0.4
        elif was_missed:
            utility_signal = 0.7

    stats["utility_score"] = (
        config.utility_momentum * stats["utility_score"]
        + (1 - config.utility_momentum) * utility_signal
    )

    persistence_signal = 1.0 if (was_useful or was_used) else 0.0
    stats["persistence_score"] = (
        config.persistence_momentum * stats["persistence_score"]
        + (1 - config.persistence_momentum) * persistence_signal
    )

    redundancy_signal = 1.0 if (was_overweighted and not was_useful) else 0.0
    stats["redundancy_score"] = 0.8 * stats["redundancy_score"] + 0.2 * redundancy_signal

    stats["retention_score"] = (
        stats["usage_score"]
        + stats["recency_score"]
        + stats["persistence_score"]
        + stats["utility_score"]
        - config.redundancy_penalty * stats["redundancy_score"]
    )

    if stats["retention_score"] < config.delete_threshold:
        stats["consecutive_low_retention"] += 1
    else:
        stats["consecutive_low_retention"] = 0

    if stats["consecutive_low_retention"] >= config.delete_after_consecutive:
        stats["status"] = "delete-candidate"
    elif stats["retention_score"] < config.weak_threshold:
        stats["status"] = "weak"
    else:
        stats["status"] = "active"


async def build_or_update_factor_memory(
    prediction_results: list[dict[str, Any]],
    existing_memory: dict[str, Any] | None = None,
    config: FactorMemoryConfig | None = None,
    trajectory_self_corrector: TrajectorySelfCorrector | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_FACTOR_MEMORY_CONFIG
    memory = copy.deepcopy(existing_memory) if existing_memory is not None else {
        "memory_type": "factor_prior_memory",
        "num_processed_results": 0,
        "factor_slots": {},
        "update_log": [],
    }
    factor_slots = memory.setdefault("factor_slots", {})
    update_log = memory.setdefault("update_log", [])

    for result_idx, prediction_result in enumerate(prediction_results, start=1):
        self_correct_result = await self_correct_trajectory(
            prediction_result=prediction_result,
            trajectory_self_corrector=trajectory_self_corrector,
        )
        hindsight_summary = summarize_factor_hindsight_from_self_correct(self_correct_result)
        update_order = determine_factor_update_order(hindsight_summary)

        for factor_key in update_order:
            factor_update = hindsight_summary["factor_updates"][factor_key]
            slot = factor_slots.setdefault(factor_key, _make_default_factor_slot(factor_key))
            update_factor_semantics(slot, factor_update)
            update_factor_statistics(
                slot=slot,
                factor_update=factor_update,
                step_idx=memory["num_processed_results"] + result_idx,
                config=config,
            )

        untouched_factor_keys = set(factor_slots) - set(update_order)
        for factor_key in untouched_factor_keys:
            update_factor_statistics(
                slot=factor_slots[factor_key],
                factor_update=None,
                step_idx=memory["num_processed_results"] + result_idx,
                config=config,
            )

        update_log.append(
            {
                "step": memory["num_processed_results"] + result_idx,
                "question": prediction_result.get("question"),
                "updated_factors": update_order,
                "useful_factors": sorted(hindsight_summary.get("useful_factor_keys", [])),
                "missed_factors": sorted(hindsight_summary.get("missed_factor_keys", [])),
                "overweighted_factors": sorted(hindsight_summary.get("overweighted_factor_keys", [])),
                "correction_note": hindsight_summary.get("correction_note", ""),
                "self_correct_result": self_correct_result,
            }
        )

    deletable = [
        factor_key
        for factor_key, slot in factor_slots.items()
        if slot.get("statistics", {}).get("status") == "delete-candidate"
    ]
    for factor_key in deletable:
        del factor_slots[factor_key]

    memory["num_processed_results"] += len(prediction_results)
    memory["factor_slots"] = dict(sorted(factor_slots.items()))
    return memory


# ---------------------------------------------------------------------------
# Retrieval / lookup: matching decomposed task factors against stored memory
# ---------------------------------------------------------------------------

def _clean_factor_label(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text)
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if " - " in text:
        prefix, suffix = text.split(" - ", 1)
        if prefix and len(prefix.split()) <= 8:
            text = prefix.strip()
        else:
            text = suffix.strip()
    return text.strip()


def extract_factor_keys_from_decomposition(decomposition: Any) -> list[str]:
    """Extract canonical factor keys from a decomposition result."""
    raw_items: list[Any] = []

    if isinstance(decomposition, dict):
        raw_items = list(
            decomposition.get("factors")
            or decomposition.get("factor_candidates")
            or decomposition.get("decomposed_factors")
            or []
        )
    elif isinstance(decomposition, (list, tuple)):
        raw_items = list(decomposition)
    elif decomposition is not None:
        raw_items = [
            line
            for line in str(decomposition).splitlines()
            if _clean_factor_label(line)
        ]

    factor_keys: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            factor_name = (
                item.get("factor_name")
                or item.get("factor")
                or item.get("name")
                or item.get("label")
            )
        else:
            factor_name = item

        cleaned = _clean_factor_label(factor_name)
        if not cleaned:
            continue

        factor_key = canonicalize_factor_name(cleaned)
        if factor_key in seen:
            continue
        seen.add(factor_key)
        factor_keys.append(factor_key)

    return factor_keys


def extract_factor_queries_from_decomposition(decomposition: Any) -> list[dict[str, str]]:
    """Extract factor query text from a decomposition result for semantic retrieval."""
    raw_items: list[Any] = []

    if isinstance(decomposition, dict):
        raw_items = list(
            decomposition.get("factors")
            or decomposition.get("factor_candidates")
            or decomposition.get("decomposed_factors")
            or []
        )
    elif isinstance(decomposition, (list, tuple)):
        raw_items = list(decomposition)
    elif decomposition is not None:
        raw_items = [
            {"factor_name": line, "rationale": ""}
            for line in str(decomposition).splitlines()
            if _clean_factor_label(line)
        ]

    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            factor_name = (
                item.get("factor_name")
                or item.get("factor")
                or item.get("name")
                or item.get("label")
            )
            rationale = (
                item.get("rationale")
                or item.get("reason")
                or item.get("summary")
                or item.get("notes")
                or ""
            )
        else:
            factor_name = item
            rationale = ""

        cleaned_name = _clean_factor_label(factor_name)
        if not cleaned_name:
            continue

        factor_key = canonicalize_factor_name(cleaned_name)
        if factor_key in seen:
            continue
        seen.add(factor_key)

        rationale_text = ""
        if isinstance(rationale, list):
            rationale_text = "; ".join(_coerce_text_list(rationale))
        else:
            rationale_text = str(rationale).strip()

        query_text = cleaned_name
        if rationale_text:
            query_text += f". {rationale_text}"

        queries.append(
            {
                "factor_key": factor_key,
                "factor_name": cleaned_name,
                "query_text": query_text.strip(),
            }
        )

    return queries


def _build_slot_retrieval_text(slot: dict[str, Any]) -> str:
    parts = [
        f"Factor: {slot.get('factor_name', slot.get('factor_key', ''))}",
        f"Meaning: {slot.get('meaning', '')}",
        f"Typical role: {slot.get('typical_role', '')}",
        f"Typical effect: {slot.get('typical_effect', '')}",
        f"Failure pattern: {slot.get('common_failure_pattern', '')}",
    ]

    update_cues = _coerce_text_list(slot.get("typical_update_cues", []))
    if update_cues:
        parts.append(f"Update cues: {'; '.join(update_cues[:8])}")

    interactions = _coerce_text_list(slot.get("common_interactions", []))
    if interactions:
        parts.append(f"Interactions: {'; '.join(interactions[:8])}")

    return "\n".join(part for part in parts if str(part).strip())


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


async def retrieve_factor_memory_from_decomposition(
    factor_memory: dict[str, Any] | None,
    decomposition: Any,
    *,
    include_weak: bool = False,
    embedding_model: str = DEFAULT_FACTOR_MEMORY_EMBEDDING_MODEL,
    max_matches: int = 5,
    min_similarity: float = 0.20,
) -> list[dict[str, Any]]:
    """Retrieve relevant memory slots for factors using embedding similarity."""
    if factor_memory is None:
        return []

    factor_slots = factor_memory.get("factor_slots", {})
    query_specs = extract_factor_queries_from_decomposition(decomposition)
    if not factor_slots or not query_specs:
        return []

    candidate_slots: list[dict[str, Any]] = []
    slot_texts: list[str] = []
    for _, slot in factor_slots.items():
        if slot is None:
            continue

        status = slot.get("statistics", {}).get("status", "active")
        if status == "delete-candidate":
            continue
        if status == "weak" and not include_weak:
            continue

        candidate_slots.append(slot)
        slot_texts.append(_build_slot_retrieval_text(slot))

    if not candidate_slots:
        return []

    _endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    _api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=_api_key, base_url=f"{_endpoint}/openai/v1/") if _endpoint and _api_key else AsyncOpenAI()
    query_embedding_response = await client.embeddings.create(
        model=embedding_model,
        input=[spec["query_text"] for spec in query_specs],
    )
    slot_embedding_response = await client.embeddings.create(
        model=embedding_model,
        input=slot_texts,
    )

    query_embeddings = [list(item.embedding) for item in query_embedding_response.data]
    slot_embeddings = [list(item.embedding) for item in slot_embedding_response.data]

    ranked_matches: list[tuple[float, dict[str, Any], dict[str, str]]] = []
    for query_spec, query_embedding in zip(query_specs, query_embeddings):
        for slot, slot_embedding in zip(candidate_slots, slot_embeddings):
            similarity = _cosine_similarity(query_embedding, slot_embedding)
            stats = slot.get("statistics", {})
            retention_score = float(stats.get("retention_score", 0.0) or 0.0)
            blended_score = similarity + 0.02 * retention_score
            if blended_score < min_similarity:
                continue
            ranked_matches.append((blended_score, slot, query_spec))

    ranked_matches.sort(
        key=lambda item: (
            -item[0],
            -float(item[1].get("statistics", {}).get("retention_score", 0.0) or 0.0),
            str(item[1].get("factor_key", "")),
        )
    )

    retrieved: list[dict[str, Any]] = []
    seen_factor_keys: set[str] = set()
    for blended_score, slot, query_spec in ranked_matches:
        factor_key = str(slot.get("factor_key", ""))
        if not factor_key or factor_key in seen_factor_keys:
            continue

        slot_copy = copy.deepcopy(slot)
        slot_copy["retrieval_score"] = round(blended_score, 4)
        slot_copy["matched_decomposition_factor"] = query_spec["factor_name"]
        slot_copy["matched_decomposition_factor_key"] = query_spec["factor_key"]
        retrieved.append(slot_copy)
        seen_factor_keys.add(factor_key)

        if len(retrieved) >= max_matches:
            break

    return retrieved


def build_factor_memory_context(retrieved_slots: list[dict[str, Any]]) -> str:
    """Format retrieved factor memory into a prompt-ready context string."""
    if not retrieved_slots:
        return ""

    sections = [
        "Relevant prior factor memory from earlier tasks:",
        "Use this as a reusable prior, not as ground truth. Prefer current task evidence when there is a conflict.",
    ]

    for slot in retrieved_slots:
        stats = slot.get("statistics", {})
        lines = [
            f"Factor: {slot.get('factor_name', slot.get('factor_key', 'Unknown'))}",
            f"Meaning: {slot.get('meaning', '')}".strip(),
            f"Typical role: {slot.get('typical_role', '')}".strip(),
        ]

        matched_factor = str(slot.get("matched_decomposition_factor", "")).strip()
        if matched_factor:
            lines.append(f"Matched decomposed factor: {matched_factor}")

        retrieval_score = slot.get("retrieval_score")
        if isinstance(retrieval_score, (int, float)):
            lines.append(f"Retrieval score: {float(retrieval_score):.4f}")

        update_cues = _coerce_text_list(slot.get("typical_update_cues", []))
        if update_cues:
            lines.append(f"Typical update cues: {'; '.join(update_cues[:5])}")

        typical_effect = str(slot.get("typical_effect", "")).strip()
        if typical_effect:
            lines.append(f"Typical effect: {typical_effect}")

        interactions = _coerce_text_list(slot.get("common_interactions", []))
        if interactions:
            lines.append(f"Common interactions: {'; '.join(interactions[:4])}")

        failure_pattern = str(slot.get("common_failure_pattern", "")).strip()
        if failure_pattern:
            lines.append(f"Failure pattern: {failure_pattern}")

        status = stats.get("status")
        retention_score = stats.get("retention_score")
        if status is not None or retention_score is not None:
            retention_text = (
                f"{float(retention_score):.2f}"
                if isinstance(retention_score, (int, float))
                else "n/a"
            )
            lines.append(f"Memory status: {status or 'unknown'}; retention_score={retention_text}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)
