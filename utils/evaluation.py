import argparse
import json
from pathlib import Path

from utils.metrics import calculate_average_brier_score, calculate_brier_score, compute_ece
from utils.agent_toolkit import load_results_payload


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "prophet_arena"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def validate_output_filename(output_filename: str) -> str:
    candidate = Path(output_filename)
    if candidate.name != output_filename or output_filename in {"", ".", ".."}:
        raise ValueError("output_filename must be a file name only, not a path.")
    return output_filename


def collect_results_files(input_path: Path) -> list[Path]:
    """Collect saved weekly results files from a file or directory."""
    if input_path.is_file():
        return [input_path]

    direct_results_files = [
        input_path / "results_with_filter.jsonl",
        input_path / "results_no_filter.jsonl",
        input_path / "results_with_filter.json",
        input_path / "results_no_filter.json",
        input_path / "results.json",
    ]
    existing_direct_files = [path for path in direct_results_files if path.is_file()]
    if existing_direct_files:
        return existing_direct_files

    results_files = sorted(input_path.glob("*/results_with_filter.jsonl"))
    results_files.extend(sorted(input_path.glob("*/results_no_filter.jsonl")))
    results_files.extend(sorted(input_path.glob("*/results_with_filter.json")))
    results_files.extend(sorted(input_path.glob("*/results_no_filter.json")))
    results_files.extend(sorted(input_path.glob("*/results.json")))
    return results_files


def _collect_probs_outcomes(samples: list[dict]) -> tuple[list[float], list[int]]:
    """Flatten predictions/outcomes from a list of evaluated samples into parallel lists."""
    probs, outcomes = [], []
    for sample in samples:
        predictions = sample.get("predictions", {})
        sample_outcomes = sample.get("outcomes", {})
        if not isinstance(predictions, dict) or not isinstance(sample_outcomes, dict):
            continue
        for key in predictions:
            if key in sample_outcomes:
                probs.append(predictions[key])
                outcomes.append(sample_outcomes[key])
    return probs, outcomes


def evaluate_sample(sample: dict, sample_idx: int) -> dict:
    """Evaluate one saved sample result."""
    predictions = sample.get("predictions", {})
    outcomes = sample.get("outcomes", {})

    if not isinstance(predictions, dict) or not isinstance(outcomes, dict):
        return {
            **sample,
            "sample_idx": sample_idx,
            "evaluation_status": "error",
            "error": (
                "Each saved sample must include dict-valued "
                "'predictions' and 'outcomes'."
            ),
            "brier_score": None,
        }

    if not predictions or not outcomes:
        return {
            **sample,
            "sample_idx": sample_idx,
            "evaluation_status": "error",
            "error": "Empty predictions or outcomes.",
            "brier_score": None,
        }

    brier_score = calculate_brier_score(predictions, outcomes)
    return {
        **sample,
        "sample_idx": sample_idx,
        "evaluation_status": "success",
        "brier_score": brier_score,
    }


def evaluate_results_file(results_path: Path) -> dict:
    """Evaluate one weekly results file."""
    results_payload = load_results_payload(results_path)
    samples = results_payload.get("samples", [])

    evaluated_samples = [
        evaluate_sample(sample, sample_idx)
        for sample_idx, sample in enumerate(samples)
    ]
    successful_samples = [
        sample for sample in evaluated_samples if sample.get("brier_score") is not None
    ]
    average_brier_score = calculate_average_brier_score(successful_samples)

    probs, outcomes = _collect_probs_outcomes(successful_samples)
    ece = compute_ece(probs, outcomes) if probs else None

    evaluation_payload = {
        "results_file": str(results_path),
        "selected_week_csv": results_payload.get("selected_week_csv"),
        "week_index": results_payload.get("week_index"),
        "model": results_payload.get("model"),
        "use_close_date_filter": results_payload.get("use_close_date_filter"),
        "num_samples": len(evaluated_samples),
        "num_successful_samples": len(successful_samples),
        "num_failed_samples": len(evaluated_samples) - len(successful_samples),
        "average_brier_score": average_brier_score,
        "ece": ece,
        "samples": evaluated_samples,
    }
    return evaluation_payload


def evaluate_predictions(
    input_path: Path,
    save: bool = True,
    output_filename: str = "evaluation.json",
) -> dict:
    """Evaluate one or more saved weekly result files."""
    output_filename = validate_output_filename(output_filename)
    results_files = collect_results_files(input_path)
    if not results_files:
        raise ValueError(f"No results files found in {input_path}")

    week_evaluations = []
    for results_file in results_files:
        week_evaluation = evaluate_results_file(results_file)
        week_evaluations.append(week_evaluation)

        if save:
            save_json(results_file.parent / output_filename, week_evaluation)

    week_average_scores = [
        {"brier_score": week["average_brier_score"]}
        for week in week_evaluations
        if week.get("average_brier_score") is not None
    ]
    overall_average_brier_score = calculate_average_brier_score(week_average_scores)

    all_probs, all_outcomes = [], []
    for week in week_evaluations:
        wp, wo = _collect_probs_outcomes(
            [s for s in week.get("samples", []) if s.get("brier_score") is not None]
        )
        all_probs.extend(wp)
        all_outcomes.extend(wo)
    overall_ece = compute_ece(all_probs, all_outcomes) if all_probs else None

    aggregate_payload = {
        "input_path": str(input_path),
        "num_weeks": len(week_evaluations),
        "average_brier_score": overall_average_brier_score,
        "ece": overall_ece,
        "weeks": week_evaluations,
    }

    return aggregate_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Prophet Arena prediction results."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to a specific results file. A week folder containing results_with_filter.jsonl or results_no_filter.jsonl also works.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute evaluation without writing evaluation.json.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="evaluation.json",
        help="File name to save in the same folder as the resolved input results.",
    )
    args = parser.parse_args()

    evaluation_payload = evaluate_predictions(
        input_path=args.input_path,
        save=not args.no_save,
        output_filename=args.output_filename,
    )

    print(
        f"Evaluated {evaluation_payload['num_weeks']} week(s). "
        f"Average Brier score: {evaluation_payload['average_brier_score']}  "
        f"ECE: {evaluation_payload['ece']}"
    )


if __name__ == "__main__":
    main()
