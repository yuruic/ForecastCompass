"""
Download futurex-ai/Futurex-Online from HuggingFace and convert to the
CSV format used by the inference pipeline.

Usage:
    python data_process/futurex_online_loader.py [--week WEEK] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "futurex_online"
HF_DATASET = "futurex-ai/Futurex-Online"


def _parse_options_from_prompt(prompt: str) -> list[str]:
    """Extract option labels (A, B, C, ...) from a prompt string."""
    return re.findall(r"\n([A-Z])\.\s+", prompt)


def _parse_option_texts_from_prompt(prompt: str) -> dict[str, str]:
    """Extract {label: text} mapping from prompt, e.g. {'A': '12 or less', ...}."""
    matches = re.findall(r"\n([A-Z])\.\s+the outcome be (.+?)(?=\n[A-Z]\.|\"|\Z)", prompt, re.DOTALL)
    return {label: text.strip() for label, text in matches}


def convert_to_pipeline_csv(
    week: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_split: str = "train",
) -> Path:
    """
    Download the HuggingFace dataset and save as a pipeline-compatible CSV.

    Returns the path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split=hf_split)
    print(f"  {len(ds)} rows loaded.")

    # Derive date range from end_time for the filename
    end_times = sorted(r["end_time"] for r in ds)
    date_start = end_times[0].replace("-", "")
    date_end = end_times[-1].replace("-", "")
    filename = f"futurex_online_week{week}_{date_start}_{date_end}.csv"
    out_path = output_dir / filename

    rows = []
    for r in ds:
        options = _parse_options_from_prompt(r["prompt"])
        if not options:
            # Fallback: try to infer from prompt structure
            options = ["Yes", "No"]

        # No ground truth available in online dataset — use empty outcome dict
        market_outcome = {opt: 0 for opt in options}

        # market_info: option label → title text (best-effort parse)
        option_texts = _parse_option_texts_from_prompt(r["prompt"])
        market_info = {
            opt: {
                "ticker": f"{r['id']}-{opt}",
                "event_ticker": r["id"],
                "title": option_texts.get(opt, opt),
            }
            for opt in options
        }

        rows.append({
            "event_ticker": r["id"],
            "title": r["en_title"],
            "original_title": r["en_title"],
            "category": "",
            "markets": json.dumps(options),
            "close_time": r["end_time"],
            "market_outcome": json.dumps(market_outcome),
            "sources": json.dumps([]),
            "market_info": json.dumps(market_info),
            "market_data": json.dumps({}),
            "snapshot_time": r["end_time"],
            "submission_id": r["id"],
            "rules": "",
            "augmented_title": r["en_title"],
            "prompt": r["prompt"],
            "level": r.get("level", ""),
            "options": json.dumps(options),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert Futurex-Online dataset.")
    parser.add_argument("--week", type=int, default=12, help="Week index to assign (default: 12)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    path = convert_to_pipeline_csv(
        week=args.week,
        output_dir=args.output_dir,
        hf_split=args.split,
    )
    print(f"Done: {path}")
