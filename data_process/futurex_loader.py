#!/usr/bin/env python3
"""
Futurex-Past Dataset Loader.

Load prediction market data from HuggingFace's Futurex-Past dataset.
Dataset: futurex-ai/Futurex-Past

This dataset provides multiple-choice forecasting questions with:
- Multiple choice options (A, B, C, ...)
- Ground truth answers
- Resolution dates
- Difficulty levels (1-4)
"""

import ast
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_options_from_prompt(prompt: str) -> Dict[str, str]:
    """
    Parse the multiple-choice options from the prompt string.

    Handles three formats:
    1. Lettered options: "A. <text>" or "A. the outcome be <text>"
    2. Boxed binary/choice: "\\boxed{Yes} or \\boxed{No}"
    3. Open-ended: "\\boxed{YOUR_PREDICTION}" -> returns empty dict (skip)

    Returns a dict mapping option key to description.
    """
    # Format 1: lettered options A. B. C. ...
    letter_pattern = r'\n([A-Z])\. (.+?)(?=\n[A-Z]\. |\"\nIMPORTANT)'
    matches = re.findall(letter_pattern, prompt, re.DOTALL)
    if matches:
        options = {}
        for letter, text in matches:
            desc = re.sub(r'^the outcome be\s+', '', text.strip())
            options[letter] = desc
        return options

    # Format 2: boxed choices — extract all \boxed{X} values (but not YOUR_PREDICTION)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed = re.findall(boxed_pattern, prompt)
    # Filter out placeholder
    choices = [c for c in boxed if c != "YOUR_PREDICTION"]
    if choices:
        return {c: c for c in choices}

    # Format 3: open-ended (YOUR_PREDICTION only) — return sentinel so caller
    # can fill markets from ground_truth
    return {"__open_ended__": True}


def _parse_ground_truth(gt_str) -> List[str]:
    """Parse ground_truth from string representation of list."""
    if isinstance(gt_str, list):
        return gt_str
    try:
        parsed = ast.literal_eval(gt_str)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def load_futurex_dataset(cache_dir: str = None) -> pd.DataFrame:
    """
    Load the Futurex-Past dataset from HuggingFace.

    Args:
        cache_dir: Directory to cache the dataset

    Returns:
        DataFrame with parsed columns
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info("Loading Futurex-Past dataset from HuggingFace...")

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset("futurex-ai/Futurex-Past", **kwargs)
    df = ds["train"].to_pandas()

    # Parse datetime
    df["end_time_dt"] = pd.to_datetime(df["end_time"], format="mixed")

    # Parse ground_truth
    df["ground_truth_parsed"] = df["ground_truth"].apply(_parse_ground_truth)

    # Parse options from prompt
    df["options"] = df["prompt"].apply(_parse_options_from_prompt)

    logger.info(f"Loaded {len(df)} events from Futurex-Past")
    logger.info(
        f"Date range: {df['end_time_dt'].min()} to {df['end_time_dt'].max()}"
    )

    return df


def format_futurex_event(row: pd.Series) -> Optional[Dict]:
    """
    Format a Futurex-Past row into a prophet_arena-compatible event dict.

    Args:
        row: DataFrame row

    Returns:
        Event dictionary compatible with the pipeline
    """
    try:
        options: Dict = row["options"]  # {letter: description}, {"Yes":"Yes",...}, or {"__open_ended__": True}
        ground_truth: List[str] = row["ground_truth_parsed"]  # ["A", "C", ...] or ["Yes"] or ["answer text"]

        is_open_ended = options.get("__open_ended__") is True

        if is_open_ended:
            # Use ground_truth answers as the markets; all are correct
            markets = ground_truth if ground_truth else ["(unknown)"]
            market_outcome = {opt: 1 for opt in markets}
            options = {opt: opt for opt in markets}
        else:
            markets = list(options.keys())  # ["A", "B", ...] or ["Yes", "No"] or ["TeamA", "TeamB"]
            # market_outcome: {option: 1 if correct else 0}
            market_outcome = {
                opt: 1 if opt in ground_truth else 0 for opt in markets
            }

        # market_info: basic metadata per option
        market_info = {
            opt: {
                "ticker": f"{row['id']}-{opt}",
                "event_ticker": row["id"],
                "market_type": "binary",
                "title": options[opt],
                "yes_ask": 50,
                "yes_bid": 50,
                "no_ask": 50,
                "no_bid": 50,
                "liquidity": 0,
                "result": "yes" if market_outcome.get(opt, 0) == 1 else "no",
            }
            for opt in markets
        }

        close_time = row["end_time_dt"].isoformat()

        event = {
            "event_ticker": row["id"],
            "title": row["title"],
            "original_title": row["title"],
            "category": f"level_{row['level']}",
            "markets": json.dumps(markets),
            "close_time": close_time,
            "market_outcome": json.dumps(market_outcome),
            "sources": json.dumps([]),
            "market_info": json.dumps(market_info),
            "market_data": json.dumps({}),
            "snapshot_time": close_time,
            "submission_id": row["id"],
            "rules": "",
            "augmented_title": row["title"],
            # Futurex-specific
            "prompt": row["prompt"],
            "level": row["level"],
            "options": json.dumps(options),
        }

        return event

    except Exception as e:
        logger.error(f"Error formatting event {row.get('id', 'unknown')}: {e}")
        return None


def collect_futurex_weekly(
    output_dir: str = None,
    cache_dir: str = None,
) -> List[str]:
    """
    Split the Futurex-Past dataset into weekly CSVs and save them.

    Weeks are numbered from oldest (week1) to newest, mirroring how
    prophet_arena files are named.

    Args:
        output_dir: Directory to save CSV files (default: data/futurex/)
        cache_dir: Directory for HuggingFace dataset cache

    Returns:
        List of paths to saved CSV files
    """
    df = load_futurex_dataset(cache_dir)

    # Group by calendar week (Mon–Sun)
    df["week_period"] = df["end_time_dt"].dt.to_period("W-SUN")
    weeks = sorted(df["week_period"].unique())

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "futurex"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for week_num, week_period in enumerate(weeks, start=1):
        week_df = df[df["week_period"] == week_period].copy()

        events = []
        for _, row in week_df.iterrows():
            event = format_futurex_event(row)
            if event:
                events.append(event)

        if not events:
            logger.warning(f"No events for week {week_num} ({week_period}), skipping")
            continue

        rows_df = pd.DataFrame(events)

        start_date = week_period.start_time.strftime("%Y%m%d")
        end_date = week_period.end_time.strftime("%Y%m%d")
        output_path = output_dir / f"futurex_week{week_num}_{start_date}_{end_date}.csv"
        rows_df.to_csv(output_path, index=False)
        saved_paths.append(str(output_path))

        logger.info(f"Saved week {week_num} ({week_period}): {len(events)} events -> {output_path.name}")

    print(f"\n=== Futurex-Past Collection Summary ===")
    print(f"Total weeks: {len(weeks)}")
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['end_time_dt'].min().date()} to {df['end_time_dt'].max().date()}")
    print(f"Output dir: {output_dir}")
    for path in saved_paths:
        print(f"  {Path(path).name}")

    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Futurex-Past dataset into weekly CSVs")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: data/futurex/)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace dataset cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    paths = collect_futurex_weekly(output_dir=args.output_dir, cache_dir=args.cache_dir)
    print(f"\nSaved {len(paths)} weekly CSV files.")
