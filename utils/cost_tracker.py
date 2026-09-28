from __future__ import annotations

from typing import Any

MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),  # input $0.15, output $0.60 per 1M tokens
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}

_COST_TRACKER: dict[str, Any] = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "estimated_cost_usd": 0.0,
    "by_label": {},
}


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_price, out_price) in MODEL_COSTS_PER_1M.items():
        if model_name.startswith(prefix):
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return None


def reset_cost_tracker() -> None:
    _COST_TRACKER["calls"] = 0
    _COST_TRACKER["input_tokens"] = 0
    _COST_TRACKER["output_tokens"] = 0
    _COST_TRACKER["estimated_cost_usd"] = 0.0
    _COST_TRACKER["by_label"] = {}


def record_cost(label: str, input_tokens: int, output_tokens: int, cost: float | None) -> None:
    _COST_TRACKER["calls"] += 1
    _COST_TRACKER["input_tokens"] += input_tokens
    _COST_TRACKER["output_tokens"] += output_tokens
    if cost is not None:
        _COST_TRACKER["estimated_cost_usd"] += cost

    label_bucket = _COST_TRACKER["by_label"].setdefault(
        label,
        {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )
    label_bucket["calls"] += 1
    label_bucket["input_tokens"] += input_tokens
    label_bucket["output_tokens"] += output_tokens
    if cost is not None:
        label_bucket["estimated_cost_usd"] += cost


def get_cost_summary() -> dict[str, Any]:
    by_label = {
        label: {
            **stats,
            "estimated_cost_usd": round(float(stats["estimated_cost_usd"]), 6),
        }
        for label, stats in sorted(
            _COST_TRACKER["by_label"].items(),
            key=lambda item: item[1]["estimated_cost_usd"],
            reverse=True,
        )
    }
    return {
        "calls": int(_COST_TRACKER["calls"]),
        "input_tokens": int(_COST_TRACKER["input_tokens"]),
        "output_tokens": int(_COST_TRACKER["output_tokens"]),
        "estimated_cost_usd": round(float(_COST_TRACKER["estimated_cost_usd"]), 6),
        "by_label": by_label,
    }
