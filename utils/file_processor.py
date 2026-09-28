import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.data_loader import format_task_for_display, load_prediction_data
from data_process.prophet_arena_loader import collect_prophet_arena_weekly, get_prophet_arena_week_count


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "prophet_arena"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prophet_arena"
RESULTS_DIR = PROJECT_ROOT / "results" / "prophet_arena"
FUTUREX_OUTPUT_DIR = PROJECT_ROOT / "data" / "futurex"
FUTUREX_RESULTS_DIR = PROJECT_ROOT / "results" / "futurex"
FUTUREX_ONLINE_OUTPUT_DIR = PROJECT_ROOT / "data" / "futurex_online"
FUTUREX_ONLINE_RESULTS_DIR = PROJECT_ROOT / "results" / "futurex_online"
USE_EXISTING_WEEKLY_FILES = True


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(obj, path, mode="continue"):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if mode not in ("replace", "continue"):
        mode = "continue"
    to_write = obj
    if mode == "continue" and os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(obj, dict):
                merged = dict(existing)
                merged.update(obj)
                to_write = merged
            elif isinstance(existing, list) and isinstance(obj, list):
                to_write = existing + obj
        except Exception:
            to_write = obj
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_write, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def load_diff_mapping(path: str) -> Dict[str, str]:
    data = read_json(path)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    raise ValueError(f"Unexpected diff file format at {path}; expected object mapping id->text")

def get_weekly_csvs(
    use_existing_weekly_files: bool = USE_EXISTING_WEEKLY_FILES,
    weeks_limit: int = 10,
    dataset: str = "prophet_arena",
) -> list[str]:
    """Read existing weekly CSVs or collect them if requested."""
    if dataset in ("futurex", "futurex_online"):
        output_dir = FUTUREX_ONLINE_OUTPUT_DIR if dataset == "futurex_online" else FUTUREX_OUTPUT_DIR
        pattern = f"{dataset}_week*.csv"
        if not use_existing_weekly_files:
            loader_script = (
                "python data_process/futurex_online_loader.py"
                if dataset == "futurex_online"
                else "python data_process/futurex_loader.py"
            )
            raise ValueError(
                f"{dataset} dataset must be pre-collected. "
                f"Run `{loader_script}` first."
            )
        weekly_csvs = sorted(
            [str(path) for path in output_dir.glob(pattern)],
            key=lambda p: int(re.search(r"week(\d+)", Path(p).name).group(1)),
            reverse=True,
        )[:weeks_limit]
        if not weekly_csvs:
            raise ValueError(
                f"No weekly CSV files found in {output_dir}. "
                f"Run `python data_process/{'futurex_online_loader' if dataset == 'futurex_online' else 'futurex_loader'}.py` to generate them."
            )
        print(f"Loaded {len(weekly_csvs)} {dataset} weekly CSV files from {output_dir}")
        for csv_path in weekly_csvs:
            print(Path(csv_path).name)
        return weekly_csvs

    # prophet_arena (default)
    available_weeks, earliest_date, latest_date = get_prophet_arena_week_count(
        cache_dir=str(CACHE_DIR)
    )
    weeks_to_use = min(weeks_limit, available_weeks)

    print(
        f"Prophet Arena coverage: {available_weeks} weeks "
        f"({earliest_date.date()} -> {latest_date.date()})"
    )

    if use_existing_weekly_files:
        pattern = "prophet_arena_week*.csv"
        weekly_csvs = sorted(
            [str(path) for path in OUTPUT_DIR.glob(pattern)],
            key=lambda p: int(re.search(r"week(\d+)", Path(p).name).group(1)),
            reverse=True,
        )[:weeks_to_use]

        if not weekly_csvs:
            raise ValueError(
                f"No weekly CSV files found in {OUTPUT_DIR}. "
                "Set use_existing_weekly_files=False to collect them."
            )

        print(f"Loaded {len(weekly_csvs)} weekly CSV files from {OUTPUT_DIR}")
        for csv_path in weekly_csvs:
            print(Path(csv_path).name)
        return weekly_csvs

    print(f"Collecting {weeks_to_use} weekly files...")
    weekly_csvs = []
    for week in range(weeks_to_use, 0, -1):
        weekly_csv = collect_prophet_arena_weekly(
            weeks_back=week,
            week_duration=1,
            time_unit="weeks",
            output_dir=str(OUTPUT_DIR),
            min_markets=1,
            max_events=500,
            cache_dir=str(CACHE_DIR),
        )
        weekly_csvs.append(weekly_csv)
        print(f"week_{week}: {weekly_csv}")

    print(f"Saved {len(weekly_csvs)} weekly CSV files.")
    return weekly_csvs


def load_weekly_tasks(selected_week_csv: str, first_k: int | None = None) -> list[dict]:
    """Load and display tasks from a weekly CSV file."""
    selected_week_df = pd.read_csv(selected_week_csv)
    if first_k is not None and first_k <= 0:
        raise ValueError("--first-k must be a positive integer.")

    weekly_tasks = load_prediction_data(
        selected_week_csv,
        end_idx=first_k,
    )

    print(f"Selected weekly CSV: {selected_week_csv}")
    print(f"CSV rows: {len(selected_week_df)}")
    print(f"Pipeline tasks loaded: {len(weekly_tasks)}")
    if first_k is not None:
        print(f"Running only the first {first_k} row(s) from this CSV.")

    if weekly_tasks:
        print()
        print(format_task_for_display(weekly_tasks[0]))
    else:
        print("No tasks were loaded from the selected Prophet Arena weekly CSV.")

    return weekly_tasks


def get_week_output_dir(
    selected_week_csv: str,
    model_name: str,
    search_provider: str,
    use_close_date_filter: bool = True,
    dataset: str = "prophet_arena",
) -> Path:
    """Return the output directory for one weekly CSV."""
    if dataset == "futurex_online":
        results_dir = FUTUREX_ONLINE_RESULTS_DIR
    elif dataset == "futurex":
        results_dir = FUTUREX_RESULTS_DIR
    else:
        results_dir = RESULTS_DIR
    csv_stem = Path(selected_week_csv).stem
    match = re.search(r"week(\d+)", csv_stem, re.IGNORECASE)
    week_name = f"week{match.group(1)}" if match else csv_stem
    week_output_dir = results_dir / model_name / search_provider / week_name
    week_output_dir.mkdir(parents=True, exist_ok=True)
    return week_output_dir


def slugify_filename(value: str, max_length: int = 60) -> str:
    """Create a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not slug:
        slug = "custom_question"
    return slug[:max_length]


def get_custom_output_dir(
    question: str,
    model_name: str,
    search_provider: str,
    use_close_date_filter: bool = True,
) -> Path:
    """Return the output directory for a custom question run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_slug = slugify_filename(question)
    output_dir = RESULTS_DIR / model_name / search_provider / "custom" / f"{timestamp}_{question_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def remove_existing_file(path: Path) -> None:
    """Delete an existing file before saving a new one."""
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Expected a file path but found non-file path: {path}")
        path.unlink()


def clear_existing_output_files(*paths: Path) -> None:
    """Delete existing output files once before starting a run."""
    for path in paths:
        remove_existing_file(path)


def save_custom_result(result_payload: dict, output_dir: Path) -> Path:
    """Save one custom question result as JSON."""
    output_path = output_dir / "result.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=True)
    return output_path


def get_results_filename(use_close_date_filter: bool) -> str:
    """Return the standard results filename for a filtering mode."""
    return "results_with_filter.jsonl" if use_close_date_filter else "results_no_filter.jsonl"


def resolve_results_path(output_dir: Path, use_close_date_filter: bool) -> Path:
    """Return the expected results path for a filtering mode."""
    return output_dir / get_results_filename(use_close_date_filter)


def resolve_existing_results_input_path(
    input_path: str | Path,
    prefer_filter: bool = True,
) -> Path:
    """Resolve a results input path from either a file or a week directory."""
    path = Path(input_path)
    if path.is_file():
        return path

    preferred_order = (
        [
            "results_with_filter.jsonl",
            "results_no_filter.jsonl",
            "results_with_filter.json",
            "results_no_filter.json",
            "results.json",
        ]
        if prefer_filter
        else [
            "results_no_filter.jsonl",
            "results_with_filter.jsonl",
            "results_no_filter.json",
            "results_with_filter.json",
            "results.json",
        ]
    )
    for filename in preferred_order:
        candidate = path / filename
        if candidate.is_file():
            return candidate

    return path


def save_week_results(
    results_payload: dict,
    week_output_dir: Path,
    use_close_date_filter: bool,
) -> Path:
    """Save all task results for one week as JSONL."""
    output_path = resolve_results_path(week_output_dir, use_close_date_filter)
    with output_path.open("w", encoding="utf-8") as f:
        metadata = {k: v for k, v in results_payload.items() if k != "samples"}
        f.write(json.dumps({"record_type": "metadata", **metadata}, ensure_ascii=True) + "\n")
        for sample in results_payload.get("samples", []):
            f.write(json.dumps({"record_type": "sample", **sample}, ensure_ascii=True) + "\n")
    return output_path

def load_results_payload(results_path: str | Path) -> dict[str, Any]:
    """Load a saved weekly results payload from disk."""
    results_path = resolve_existing_results_input_path(results_path)
    if results_path.suffix == ".jsonl":
        metadata: dict[str, Any] = {}
        samples: list[dict[str, Any]] = []
        with results_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.pop("record_type", None)
                if record_type == "metadata":
                    metadata.update(record)
                elif record_type == "sample":
                    samples.append(record)
                elif record_type is None:
                    # Fallback: infer type from content
                    if "question" in record or "predictions" in record or "status" in record:
                        samples.append(record)
                    else:
                        metadata.update(record)
        metadata["samples"] = samples
        metadata["num_samples"] = len(samples)
        return metadata

    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)
