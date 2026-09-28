from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import openai
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from agents import set_default_openai_client, set_tracing_disabled
from utils.cost_tracker import (
    estimate_cost,
    get_cost_summary,
    record_cost,
    reset_cost_tracker,
)
from utils.agent_toolkit import (
    build_week_results_payload,
    load_factor_memory,
    load_results_payload,
    load_weekly_tasks,
    resolve_existing_results_input_path,
    resolve_results_path,
    save_week_results,
)
from utils.inference_runner import run_single_prediction
from utils.evaluation import evaluate_predictions
from utils.data_loader import create_prediction_prompt


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompt"
INIT_CTGR_DIR = PROJECT_ROOT / "init_ctgr"
DEFAULT_TAXONOMY_PATH = INIT_CTGR_DIR / "prophet_arena_ctgr.json"


def get_default_taxonomy_path(dataset: str = "prophet_arena") -> Path:
    """Return the initial category file for the given dataset."""
    return INIT_CTGR_DIR / f"{dataset}_ctgr.json"
CLASSIFICATION_PROMPT_PATH = PROMPT_DIR / "question_category_classification.txt"
VERIFY_PROMPT_PATH = PROMPT_DIR / "verify_new_category_proposal.txt"
SUBCATEGORY_MEMORY_PROMPT_PATH = PROMPT_DIR / "memory_from_trajectory.txt"
REVISE_MEMORY_PROMPT_PATH = PROMPT_DIR / "revise_subcategory_memory_from_epochs.txt"
EVENT_MEMORY_REVISION_SUGGESTION_PROMPT_PATH = (
    PROMPT_DIR / "suggest_subcategory_memory_revision_for_event.txt"
)
BATCH_MERGE_SUGGESTION_PROMPT_PATH = PROMPT_DIR / "merge_batch_revision_suggestions.txt"

# Basic file and prompt helpers.

def _make_openai_client(provider: str = "azure") -> AsyncOpenAI:
    """Create an AsyncOpenAI client.

    provider: "auto" (env-var detection), "azure", "openai", or "gemini".
    """
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    use_azure = False
    if provider == "azure":
        use_azure = True
    elif provider == "auto":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        use_azure = bool(endpoint and api_key)

    if use_azure:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        base_url = endpoint.rstrip("/") + "/openai/v1/"
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI()


def _chat_temperature(model: str) -> float | None:
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return None
    return 0.7


def _chat_top_p(model: str) -> float | None:
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return None
    return 0.7


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return path


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_question_text(question: Any) -> str:
    return " ".join(str(question or "").split()).strip()


def _taxonomy_categories(taxonomy_payload: dict[str, Any]) -> list[dict[str, Any]]:
    categories = taxonomy_payload.get("categories", [])
    if not isinstance(categories, list):
        raise TypeError("Taxonomy payload must contain a list-valued 'categories' field.")
    return categories


def _taxonomy_text(taxonomy_payload: dict[str, Any]) -> str:
    """Return taxonomy as JSON with braces escaped for use in str.format() templates."""
    raw = json.dumps(taxonomy_payload, ensure_ascii=False, indent=2)
    return raw.replace("{", "{{").replace("}", "}}")


def _sanitize_str(value: Any) -> Any:
    """Recursively strip control characters from strings in a JSON-serializable value.

    Null bytes and other control characters (U+0000–U+001F, except tab/newline/CR)
    cause OpenAI to reject the request body as invalid JSON.
    """
    if isinstance(value, str):
        return "".join(ch for ch in value if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 0x20)
    if isinstance(value, dict):
        return {k: _sanitize_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_str(item) for item in value]
    return value


def _fmt_json(value: Any) -> str:
    """Serialize value to JSON and escape braces so it is safe inside str.format() templates."""
    raw = json.dumps(_sanitize_str(value), ensure_ascii=False, indent=2)
    return raw.replace("{", "{{").replace("}", "}}")


async def _chat_json(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 120,
    label: str = "memory-llm",
) -> dict[str, Any]:
    create_kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": timeout,
    }
    temperature = _chat_temperature(model)
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    top_p = _chat_top_p(model)
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**create_kwargs)
            break
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"[{label}] rate limit hit, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
    usage = response.usage
    if usage is not None:
        in_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens
        cost = estimate_cost(model, in_tok, out_tok)
        record_cost(label, in_tok, out_tok, cost)
        cost_str = f"  est. cost ${cost:.4f}" if cost is not None else ""
        print(f"[{label}] tokens: {in_tok} in / {out_tok} out{cost_str}")
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def _extract_questions_from_results(results_payload: dict[str, Any]) -> list[str]:
    return [
        sample["question"]
        for sample in results_payload.get("samples", [])
        if isinstance(sample, dict) and sample.get("question")
    ]

# Taxonomy preprocessing and merge helpers.


def _group_new_proposals(classification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in classification_rows:
        if row.get("matched_existing_taxonomy"):
            continue
        category_name = (row.get("proposed_category_name") or "").strip()
        subcategory_name = (row.get("proposed_subcategory_name") or "").strip()
        if not category_name or not subcategory_name:
            continue

        key = (category_name, subcategory_name)
        group = grouped.setdefault(
            key,
            {
                "proposed_category_name": category_name,
                "proposed_subcategory_name": subcategory_name,
                "count": 0,
                "example_questions": [],
                "rationales": [],
                "avg_confidence": 0.0,
            },
        )
        group["count"] += 1
        confidence = float(row.get("confidence") or 0.0)
        group["avg_confidence"] += confidence
        if row.get("question") and len(group["example_questions"]) < 5:
            group["example_questions"].append(row["question"])
        rationale = row.get("proposed_category_rationale") or row.get("rationale")
        if rationale and len(group["rationales"]) < 5:
            group["rationales"].append(rationale)

    grouped_rows = []
    for group in grouped.values():
        if group["count"] > 0:
            group["avg_confidence"] = round(group["avg_confidence"] / group["count"], 4)
        grouped_rows.append(group)

    grouped_rows.sort(
        key=lambda row: (
            -int(row["count"]),
            -float(row["avg_confidence"]),
            row["proposed_category_name"],
            row["proposed_subcategory_name"],
        )
    )
    return grouped_rows


def _dedupe_texts(values: list[str], max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _merge_verified_proposals(
    taxonomy_payload: dict[str, Any],
    verified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = copy.deepcopy(taxonomy_payload)
    categories = _taxonomy_categories(merged)

    category_lookup = {
        str(category.get("category_name")): category
        for category in categories
        if category.get("category_name")
    }

    for row in verified_rows:
        if not row.get("promote_to_taxonomy"):
            continue

        target_category_name = row.get("target_category_name")
        target_subcategory_name = row.get("target_subcategory_name")
        if not target_category_name or not target_subcategory_name:
            continue

        category = category_lookup.get(target_category_name)
        if category is None:
            category = {
                "category_name": target_category_name,
                "summarized_patterns": _dedupe_texts(
                    [str(item) for item in row.get("category_summarized_patterns", [])],
                    max_items=3,
                ),
                "examples": _dedupe_texts(
                    [str(item) for item in row.get("examples", [])],
                    max_items=3,
                ),
                "subcategories": [],
            }
            categories.append(category)
            category_lookup[target_category_name] = category
        else:
            category["summarized_patterns"] = _dedupe_texts(
                list(category.get("summarized_patterns", []))
                + [str(item) for item in row.get("category_summarized_patterns", [])],
                max_items=3,
            )
            category["examples"] = _dedupe_texts(
                list(category.get("examples", []))
                + [str(item) for item in row.get("examples", [])],
                max_items=3,
            )

        subcategories = category.setdefault("subcategories", [])
        subcategory_lookup = {
            str(subcategory.get("subcategory_name")): subcategory
            for subcategory in subcategories
            if subcategory.get("subcategory_name")
        }
        subcategory = subcategory_lookup.get(target_subcategory_name)
        if subcategory is None:
            subcategories.append(
                {
                    "subcategory_name": target_subcategory_name,
                    "summarized_patterns": _dedupe_texts(
                        [str(item) for item in row.get("subcategory_summarized_patterns", [])],
                        max_items=3,
                    ),
                    "examples": _dedupe_texts(
                        [str(item) for item in row.get("examples", [])],
                        max_items=3,
                    ),
                }
            )
        else:
            subcategory["summarized_patterns"] = _dedupe_texts(
                list(subcategory.get("summarized_patterns", []))
                + [str(item) for item in row.get("subcategory_summarized_patterns", [])],
                max_items=3,
            )
            subcategory["examples"] = _dedupe_texts(
                list(subcategory.get("examples", []))
                + [str(item) for item in row.get("examples", [])],
                max_items=3,
            )

    categories.sort(key=lambda category: str(category.get("category_name", "")))
    for category in categories:
        category["subcategories"] = sorted(
            category.get("subcategories", []),
            key=lambda subcategory: str(subcategory.get("subcategory_name", "")),
        )
    return merged


async def preprocess(
    *,
    results_path: str | Path,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    classification_output_path: str | Path | None = None,
    grouped_proposals_output_path: str | Path | None = None,
    verified_proposals_output_path: str | Path | None = None,
    taxonomy_backup_path: str | Path | None = None,
    taxonomy_output_path: str | Path | None = None,
    model_name: str = "gpt-5-mini",
    overwrite: bool = False,
    max_classification_concurrency: int = 10,
    skip_classification: bool = False,
    update_taxonomy: bool = True,
    provider: str = "azure",
) -> dict[str, Any]:
    results_path = Path(results_path)
    base_dir = results_path.parent
    classification_output_path = Path(
        classification_output_path or base_dir / "question_category_classification.json"
    )
    grouped_proposals_output_path = Path(
        grouped_proposals_output_path or base_dir / "proposed_new_category_groups.json"
    )
    verified_proposals_output_path = Path(
        verified_proposals_output_path or base_dir / "verified_new_category_proposals.json"
    )
    taxonomy_path = Path(taxonomy_path)
    taxonomy_backup_path = Path(
        taxonomy_backup_path or base_dir / (taxonomy_path.stem + "_before_merge.json")
    )

    results_payload = load_results_payload(results_path)
    questions = _extract_questions_from_results(results_payload)
    taxonomy_payload = _load_json(taxonomy_path)

    client = _make_openai_client(provider)
    classification_prompt = _load_text(CLASSIFICATION_PROMPT_PATH)
    verify_prompt = _load_text(VERIFY_PROMPT_PATH)

    if skip_classification:
        if not classification_output_path.exists():
            raise FileNotFoundError(
                f"--skip-classification requires an existing classification file at "
                f"{classification_output_path}, but none was found."
            )
        print(f"Skipping classification — loading from {classification_output_path}", flush=True)
        classification_rows = _load_json(classification_output_path)
    elif classification_output_path.exists() and not overwrite:
        print(f"Loading cached classification from {classification_output_path}", flush=True)
        classification_rows = _load_json(classification_output_path)
    else:
        taxonomy_text = _taxonomy_text(taxonomy_payload)
        semaphore = asyncio.Semaphore(max(1, max_classification_concurrency))
        checkpoint_path = classification_output_path.with_suffix(".partial.jsonl")

        # Resume from partial checkpoint if present (from a prior interrupted run)
        completed: dict[str, dict[str, Any]] = {}
        if checkpoint_path.exists() and not overwrite:
            with checkpoint_path.open("r", encoding="utf-8") as _ckpt_f:
                for _line in _ckpt_f:
                    _line = _line.strip()
                    if _line:
                        try:
                            _row = json.loads(_line)
                            if _row.get("question"):
                                completed[_row["question"]] = _row
                        except json.JSONDecodeError:
                            pass
            if completed:
                print(
                    f"Resuming from checkpoint: {len(completed)}/{len(questions)} already classified",
                    flush=True,
                )

        pending_questions = [q for q in questions if q not in completed]
        pbar = tqdm(total=len(questions), initial=len(completed), desc="Classifying questions", unit="q")
        checkpoint_lock = asyncio.Lock()

        async def _classify_one(question: str) -> dict[str, Any]:
            async with semaphore:
                user_prompt = classification_prompt.format(
                    taxonomy_text=taxonomy_text,
                    question=question,
                )
                parsed = await _chat_json(
                    client,
                    model=model_name,
                    system_prompt=(
                        "Classify forecasting questions into the provided taxonomy. "
                        "Return valid JSON only."
                    ),
                    user_prompt=user_prompt,
                    label=f"classify [{question[:40]}]",
                )
                parsed["question"] = question
                async with checkpoint_lock:
                    _append_jsonl(checkpoint_path, [parsed])
                pbar.update(1)
                return parsed

        new_rows = await asyncio.gather(
            *[_classify_one(q) for q in pending_questions]
        )
        pbar.close()

        all_rows = dict(completed)
        for row in new_rows:
            all_rows[row["question"]] = row
        classification_rows = [all_rows[q] for q in questions if q in all_rows]
        _save_json(classification_output_path, classification_rows)
        checkpoint_path.unlink(missing_ok=True)

    grouped_rows = _group_new_proposals(classification_rows)
    _save_json(grouped_proposals_output_path, grouped_rows)

    verified_rows: list[dict[str, Any]] = []
    if update_taxonomy and grouped_rows:
        taxonomy_text = _taxonomy_text(taxonomy_payload)
        for proposal in grouped_rows:
            user_prompt = verify_prompt.format(
                taxonomy_text=taxonomy_text,
                proposal_json=json.dumps(proposal, ensure_ascii=False, indent=2),
            )
            parsed = await _chat_json(
                client,
                model=model_name,
                system_prompt=(
                    "Verify whether grouped taxonomy proposals should be merged into the "
                    "forecasting taxonomy. Return valid JSON only."
                ),
                user_prompt=user_prompt,
                label="verify-taxonomy",
            )
            parsed["proposal"] = proposal
            verified_rows.append(parsed)
        _save_json(verified_proposals_output_path, verified_rows)

        merged_taxonomy = _merge_verified_proposals(taxonomy_payload, verified_rows)

        # Back-fill classification rows for questions whose proposed category was promoted.
        # Without this, unmatched questions keep category_name=None and are silently
        # skipped during memory training, so new taxonomy entries never get training data.
        promoted_map: dict[tuple[str, str], tuple[str, str]] = {}
        for vrow in verified_rows:
            if (
                vrow.get("promote_to_taxonomy")
                and vrow.get("target_category_name")
                and vrow.get("target_subcategory_name")
            ):
                proposal = vrow.get("proposal", {})
                prop_cat = str(proposal.get("proposed_category_name") or "").strip()
                prop_sub = str(proposal.get("proposed_subcategory_name") or "").strip()
                if prop_cat and prop_sub:
                    promoted_map[(prop_cat, prop_sub)] = (
                        vrow["target_category_name"],
                        vrow["target_subcategory_name"],
                    )
        if promoted_map:
            n_updated = 0
            for cls_row in classification_rows:
                if cls_row.get("matched_existing_taxonomy"):
                    continue
                prop_cat = str(cls_row.get("proposed_category_name") or "").strip()
                prop_sub = str(cls_row.get("proposed_subcategory_name") or "").strip()
                target = promoted_map.get((prop_cat, prop_sub))
                if target:
                    cls_row["category_name"] = target[0]
                    cls_row["subcategory_name"] = target[1]
                    cls_row["matched_existing_taxonomy"] = True
                    n_updated += 1
            if n_updated:
                _save_json(classification_output_path, classification_rows)
                print(f"Back-filled {n_updated} classification rows with promoted category assignments.", flush=True)

        local_taxonomy_path = Path(taxonomy_output_path) if taxonomy_output_path else base_dir / taxonomy_path.name
        if merged_taxonomy != taxonomy_payload:
            _save_json(taxonomy_backup_path, taxonomy_payload)
        _save_json(local_taxonomy_path, merged_taxonomy)
        taxonomy_path = local_taxonomy_path
        taxonomy_payload = merged_taxonomy
    else:
        if overwrite or not verified_proposals_output_path.exists():
            _save_json(verified_proposals_output_path, verified_rows)
        local_taxonomy_path = Path(taxonomy_output_path) if taxonomy_output_path else base_dir / taxonomy_path.name
        _save_json(local_taxonomy_path, taxonomy_payload)
        taxonomy_path = local_taxonomy_path

    return {
        "classification_output_path": str(classification_output_path),
        "grouped_proposals_output_path": str(grouped_proposals_output_path),
        "verified_proposals_output_path": str(verified_proposals_output_path),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_backup_path": str(taxonomy_backup_path),
    }


async def _classify_questions(
    *,
    questions: list[str],
    taxonomy_path: str | Path,
    classification_output_path: str | Path,
    model_name: str = "gpt-5-mini",
    max_concurrency: int = 10,
    provider: str = "azure",
) -> Path:
    """Classify a list of questions against the taxonomy and save results.

    Used in csv_only_mode where no baseline results file exists (e.g. futurex_online).
    """
    classification_output_path = Path(classification_output_path)
    taxonomy_path = Path(taxonomy_path)
    taxonomy_payload = _load_json(taxonomy_path)
    taxonomy_text = _taxonomy_text(taxonomy_payload)
    classification_prompt = _load_text(CLASSIFICATION_PROMPT_PATH)
    client = _make_openai_client(provider)
    semaphore = asyncio.Semaphore(max_concurrency)

    print(f"Classifying {len(questions)} questions from CSV against taxonomy...", flush=True)

    async def _classify_one(question: str) -> dict[str, Any]:
        async with semaphore:
            user_prompt = classification_prompt.format(
                taxonomy_text=taxonomy_text,
                question=question,
            )
            parsed = await _chat_json(
                client,
                model=model_name,
                system_prompt="Classify forecasting questions into the provided taxonomy. Return valid JSON only.",
                user_prompt=user_prompt,
                label=f"classify [{question[:40]}]",
            )
            parsed["question"] = question
            return parsed

    rows = list(await asyncio.gather(*[_classify_one(q) for q in questions if q]))
    _save_json(classification_output_path, rows)
    print(f"Classification saved to {classification_output_path}", flush=True)
    return classification_output_path


async def _ensure_classification_path(
    *,
    classification_path: str | Path,
    taxonomy_path: str | Path,
    baseline_results_path: str | Path | None,
    csv_only_mode: bool,
    model_name: str | None,
    overwrite_preprocess: bool,
    output_dir: str | Path | None = None,
    provider: str = "azure",
) -> Path:
    classification_path = Path(classification_path)
    taxonomy_path = Path(taxonomy_path)

    if classification_path.exists() and not overwrite_preprocess:
        print(f"Using existing classification at {classification_path}", flush=True)
        return classification_path

    if csv_only_mode:
        if classification_path.exists() and not overwrite_preprocess:
            print(f"CSV-only mode: using cached classification at {classification_path}", flush=True)
            return classification_path
        # No baseline results file, but we can classify questions directly from the weekly CSV
        # that was passed as evaluated_data_path. The CSV path is recoverable from output_dir's
        # parent structure, but the cleanest path is to defer to the caller who has selected_week_csv.
        # We signal this by returning the (non-existent) path; the caller must handle generation.
        print(
            "CSV-only mode: classification file not found — will classify questions from the weekly CSV.",
            flush=True,
        )
        return classification_path

    if baseline_results_path is None:
        raise ValueError("Cannot generate classification without a baseline results path.")
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"taxonomy_path does not exist: {taxonomy_path}"
        )

    # Root all preprocess outputs under the specified output_dir, falling back to
    # the week folder derived from the baseline results path.
    if output_dir is not None:
        _week_dir = Path(output_dir)
    else:
        _results_parent = Path(baseline_results_path).parent
        _week_dir = _results_parent.parent if _results_parent.name == "baselines" else _results_parent

    reason = "overwrite requested" if overwrite_preprocess else "classification cache missing"
    print(
        f"Preparing classification via taxonomy {taxonomy_path} because {reason}.",
        flush=True,
    )
    preprocess_outputs = await preprocess(
        results_path=baseline_results_path,
        taxonomy_path=taxonomy_path,
        classification_output_path=classification_path,
        grouped_proposals_output_path=_week_dir / "proposed_new_category_groups.json",
        verified_proposals_output_path=_week_dir / "verified_new_category_proposals.json",
        taxonomy_backup_path=_week_dir / (Path(taxonomy_path).stem + "_before_merge.json"),
        taxonomy_output_path=_week_dir / taxonomy_path.name,
        model_name=model_name,
        overwrite=overwrite_preprocess,
        update_taxonomy=False,
        provider=provider,
    )
    return Path(preprocess_outputs["classification_output_path"])

# Trajectory extraction and memory-shaping helpers.


def _classification_lookup(
    classification_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in classification_rows:
        question = row.get("question")
        if not question:
            continue
        raw_question = str(question)
        lookup[raw_question] = row
        normalized_question = _normalize_question_text(raw_question)
        if normalized_question:
            lookup.setdefault(normalized_question, row)
    return lookup



def _load_samples_payload(results_or_evaluation_path: str | Path) -> dict[str, Any]:
    path = Path(results_or_evaluation_path)
    if path.suffix == ".jsonl":
        from utils.results_io import load_results_payload
        return load_results_payload(path)
    return _load_json(path)


def _resolve_baseline_evaluation_path(
    evaluated_data_path: str | Path,
    *,
    prefer_filter: bool = True,
) -> Path | None:
    path = Path(evaluated_data_path)
    base_dir = path if path.is_dir() else path.parent
    baselines_dir = base_dir / "baselines"

    candidates = (
        [
            baselines_dir / "evaluation_with_filter.json",
            baselines_dir / "evaluation_no_filter.json",
            base_dir / "evaluation_with_filter.json",
            base_dir / "evaluation_no_filter.json",
            base_dir / "evaluation.json",
        ]
        if prefer_filter
        else [
            baselines_dir / "evaluation_no_filter.json",
            baselines_dir / "evaluation_with_filter.json",
            base_dir / "evaluation_no_filter.json",
            base_dir / "evaluation_with_filter.json",
            base_dir / "evaluation.json",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _build_classified_sample_rows(
    payload: dict[str, Any],
    classification_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _classification_lookup(classification_rows)
    rows: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        question = sample.get("question")
        classification = lookup.get(str(question), {})
        rows.append(
            {
                "question": question,
                "ground_truth": sample.get("outcomes"),
                "prediction": sample.get("predictions"),
                "raw_answer": sample.get("raw_answer"),
                "brier_score": sample.get("brier_score"),
                "category_name": classification.get("category_name"),
                "subcategory_name": classification.get("subcategory_name"),
            }
        )
    return rows


def _group_rows_by_subcategory(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        category_name = row.get("category_name")
        subcategory_name = row.get("subcategory_name")
        if not category_name or not subcategory_name:
            continue
        grouped.setdefault((str(category_name), str(subcategory_name)), []).append(row)
    return grouped


def _build_trajectory_bundle(
    rows: list[dict[str, Any]],
    *,
    sample_limit: int,
) -> list[dict[str, Any]]:
    sampled_rows = rows[:]
    if len(sampled_rows) > sample_limit:
        random.Random(42).shuffle(sampled_rows)
        sampled_rows = sampled_rows[:sample_limit]

    return [
        {
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "prediction": row.get("prediction"),
            "raw_answer": row.get("raw_answer"),
        }
        for row in sampled_rows
    ]


def _build_single_event_bundle(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "prediction": row.get("prediction"),
        "raw_answer": row.get("raw_answer"),
    }


def _load_memory_entries(memory_path: str | Path) -> list[dict[str, Any]]:
    path = Path(memory_path)
    if not path.exists():
        return []
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise TypeError("Expected subcategory memory JSON to be a list of entries.")
    taxonomy_payload = _load_json(DEFAULT_TAXONOMY_PATH)
    subcategory_to_category: dict[str, str] = {}
    for category in _taxonomy_categories(taxonomy_payload):
        category_name = str(category.get("category_name") or "").strip()
        for subcategory in category.get("subcategories", []) or []:
            subcategory_name = str(subcategory.get("subcategory_name") or "").strip()
            if category_name and subcategory_name:
                subcategory_to_category[subcategory_name] = category_name

    enriched_payload: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        enriched_entry = copy.deepcopy(entry)
        category_name = str(enriched_entry.get("category_name") or "").strip()
        subcategory_name = str(enriched_entry.get("subcategory_name") or "").strip()
        if not category_name and subcategory_name:
            inferred_category = subcategory_to_category.get(subcategory_name)
            if inferred_category:
                enriched_entry["category_name"] = inferred_category
        enriched_payload.append(enriched_entry)
    return enriched_payload


def _index_memory_entries(
    entries: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        category_name = entry.get("category_name")
        subcategory_name = entry.get("subcategory_name")
        if category_name and subcategory_name:
            lookup[(str(category_name), str(subcategory_name))] = entry
    return lookup


def _memory_entries_have_keys(entries: list[dict[str, Any]]) -> bool:
    return all(
        isinstance(entry, dict)
        and bool(entry.get("category_name"))
        and bool(entry.get("subcategory_name"))
        for entry in entries
    )


# Convert saved memory into prompt text for inference-time retrieval.

def _normalize_common_factors(common_factors: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(common_factors, list):
        return normalized

    for factor in common_factors:
        if isinstance(factor, dict):
            factor_name = str(factor.get("factor_name") or factor.get("name") or "").strip()
            if not factor_name:
                continue
            description = str(factor.get("description") or "").strip()
            typical_effect_on_output = str(factor.get("typical_effect_on_output") or "").strip()
            common_failures_raw = factor.get("common_failures") or []
            if isinstance(common_failures_raw, list):
                common_failures = [str(item).strip() for item in common_failures_raw if str(item).strip()]
            else:
                common_failures = [str(common_failures_raw).strip()] if str(common_failures_raw).strip() else []
            normalized.append(
                {
                    "factor_name": factor_name,
                    "description": description,
                    "typical_effect_on_output": typical_effect_on_output,
                    "common_failures": common_failures,
                }
            )
        else:
            factor_name = str(factor).strip()
            if not factor_name:
                continue
            normalized.append(
                {
                    "factor_name": factor_name,
                    "description": "",
                    "typical_effect_on_output": "",
                    "common_failures": [],
                }
            )
    return normalized


def build_factor_memory_context_from_subcategory(memory_entry: dict[str, Any]) -> str:
    lines = [
        "Reusable forecasting factor memory:",
        f"Category: {memory_entry.get('category_name')}",
        f"Subcategory: {memory_entry.get('subcategory_name')}",
    ]

    common_factors = _normalize_common_factors(memory_entry.get("common_factors") or [])
    if common_factors:
        lines.append("Common factors to consider:")
        for factor in common_factors[:10]:
            lines.append(f"- {factor['factor_name']}")
            if factor.get("description"):
                lines.append(f"  Description: {factor['description']}")
            if factor.get("typical_effect_on_output"):
                lines.append(f"  Typical effect: {factor['typical_effect_on_output']}")
            reasoning_patterns = factor.get("common_reasoning_patterns") or []
            if reasoning_patterns:
                joined_patterns = "; ".join(str(item).strip() for item in reasoning_patterns[:3] if str(item).strip())
                if joined_patterns:
                    lines.append(f"  Common reasoning patterns: {joined_patterns}")
            if factor.get("common_failures"):
                joined_failures = "; ".join(factor["common_failures"][:3])
                lines.append(f"  Common failures: {joined_failures}")

    return "\n".join(lines)


def _normalize_reasoning_memory_patterns(value: Any) -> dict[str, list[str]]:
    def _str_list(items: Any) -> list[str]:
        return [str(item).strip() for item in (items or []) if str(item).strip()]

    if isinstance(value, dict):
        # Full 5-field schema
        if any(k in value for k in ("calibration_experiences", "overconfidence_patterns",
                                    "underconfidence_patterns", "probability_update_lessons",
                                    "common_reasoning_failures")):
            return {
                "calibration_experiences": _str_list(value.get("calibration_experiences")),
                "overconfidence_patterns": _str_list(value.get("overconfidence_patterns")),
                "underconfidence_patterns": _str_list(value.get("underconfidence_patterns")),
                "probability_update_lessons": _str_list(value.get("probability_update_lessons")),
                "common_reasoning_failures": _str_list(value.get("common_reasoning_failures")),
            }
        # Legacy schema: collapse into calibration_experiences + common_reasoning_failures
        calibration = _str_list(
            (value.get("common_experiences") or [])
            + (value.get("distribution_calibration_patterns") or [])
            + (value.get("common_calibration_errors") or [])
        )
        failures = _str_list(
            (value.get("common_error_patterns") or [])
            + (value.get("other_reasoning_errors") or [])
        )
        return {
            "calibration_experiences": calibration,
            "overconfidence_patterns": [],
            "underconfidence_patterns": [],
            "probability_update_lessons": [],
            "common_reasoning_failures": failures,
        }

    legacy_items = _str_list(value)
    return {
        "calibration_experiences": legacy_items,
        "overconfidence_patterns": [],
        "underconfidence_patterns": [],
        "probability_update_lessons": [],
        "common_reasoning_failures": [],
    }


def build_reasoning_memory_context_from_subcategory(memory_entry: dict[str, Any]) -> str:
    lines = [
        "Reusable forecasting reasoning memory:",
        f"Category: {memory_entry.get('category_name')}",
        f"Subcategory: {memory_entry.get('subcategory_name')}",
        "Use these as guardrails. Actively avoid repeating the listed failures and calibration mistakes.",
    ]

    common_reasoning = _normalize_reasoning_memory_patterns(
        memory_entry.get("common_reasoning_patterns", memory_entry.get("common_reasoning_procedure"))
    )
    if common_reasoning.get("calibration_experiences"):
        lines.append("Calibration experiences:")
        lines.extend(f"- {step}" for step in common_reasoning["calibration_experiences"][:8])
    if common_reasoning.get("overconfidence_patterns"):
        lines.append("Overconfidence patterns to avoid:")
        lines.extend(f"- {step}" for step in common_reasoning["overconfidence_patterns"][:8])
    if common_reasoning.get("underconfidence_patterns"):
        lines.append("Underconfidence patterns to avoid:")
        lines.extend(f"- {step}" for step in common_reasoning["underconfidence_patterns"][:8])
    if common_reasoning.get("probability_update_lessons"):
        lines.append("Probability update lessons:")
        lines.extend(f"- {step}" for step in common_reasoning["probability_update_lessons"][:8])
    if common_reasoning.get("common_reasoning_failures"):
        lines.append("Common reasoning failures to avoid:")
        lines.extend(f"- {step}" for step in common_reasoning["common_reasoning_failures"][:8])

    representative_examples = memory_entry.get("representative_examples") or []
    if representative_examples:
        lines.append("Representative examples:")
        lines.extend(f"- {example}" for example in representative_examples[:5])

    notes = str(memory_entry.get("notes") or "").strip()
    if notes:
        lines.append(f"Notes: {notes}")

    return "\n".join(lines)


def _compact_memory_entry_for_prompt(memory_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "category_name": memory_entry.get("category_name"),
        "subcategory_name": memory_entry.get("subcategory_name"),
        "num_questions": memory_entry.get("num_questions"),
        "num_sampled_questions": memory_entry.get("num_sampled_questions"),
        "common_factors": memory_entry.get("common_factors", []),
        "common_reasoning_patterns": memory_entry.get(
            "common_reasoning_patterns",
            memory_entry.get("common_reasoning_procedure", []),
        ),
        "representative_examples": memory_entry.get("representative_examples", [])[:5],
        "notes": memory_entry.get("notes", ""),
    }


def _compact_memory_history(memory_history: Any, keep_last: int = 3) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in list(memory_history or [])[-keep_last:]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "epoch": item.get("epoch"),
                "kind": item.get("kind"),
                "source_results_path": item.get("source_results_path"),
                "nofilter_results_path": item.get("nofilter_results_path"),
                "generated_at": item.get("generated_at"),
                "update_rationale": item.get("update_rationale", ""),
                "num_event_revision_suggestions": len(item.get("event_revision_suggestions") or []),
            }
        )
    return compacted


def _strip_memory_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for entry in entries:
        cleaned = copy.deepcopy(entry)
        cleaned.pop("memory_history", None)
        stripped.append(cleaned)
    return stripped


# Train/update subcategory memory from one epoch of evaluated results.

async def train(
    *,
    current_results_path: str | Path,
    classification_path: str | Path,
    memory_output_path: str | Path,
    epoch: int,
    model_name: str = "gpt-5-mini",
    nofilter_results_path: str | Path | None = None,
    snapshot_output_path: str | Path | None = None,
    revision_report_output_path: str | Path | None = None,
    sample_limit: int = 8,
    max_factors: int = 8,
    max_subcategory_concurrency: int = 3,
    suggestion_batch_size: int = 30,
    overwrite: bool = False,
    force_revise: bool = False,
    provider: str = "azure",
) -> dict[str, Any]:
    current_results_path = Path(current_results_path)
    memory_output_path = Path(memory_output_path)
    snapshot_output_path = Path(snapshot_output_path or current_results_path.parent / "memory.json")
    revision_report_output_path = Path(
        revision_report_output_path or current_results_path.parent / f"memory_revision_epoch_{epoch}.json"
    )

    if epoch == 1 and memory_output_path.exists() and not overwrite and not force_revise:
        existing_entries = _load_memory_entries(memory_output_path)
        if _memory_entries_have_keys(existing_entries):
            clean_entries = _strip_memory_history(existing_entries)
            _save_json(memory_output_path, clean_entries)
            if snapshot_output_path != memory_output_path:
                _save_json(snapshot_output_path, clean_entries)
            return {
                "memory_output_path": str(memory_output_path),
                "snapshot_output_path": str(snapshot_output_path),
                "revision_report_output_path": str(revision_report_output_path),
                "num_entries": len(clean_entries),
                "skipped": True,
                "memory_entries": clean_entries,
                "revision_rows": [],
            }

    classification_rows = _load_json(classification_path)
    current_payload = _load_samples_payload(current_results_path)
    current_rows = _build_classified_sample_rows(current_payload, classification_rows)
    current_groups = _group_rows_by_subcategory(current_rows)

    nofilter_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if nofilter_results_path is not None and Path(nofilter_results_path).exists():
        nofilter_payload = _load_samples_payload(nofilter_results_path)
        nofilter_rows = _build_classified_sample_rows(nofilter_payload, classification_rows)
        nofilter_groups = _group_rows_by_subcategory(nofilter_rows)

    client = _make_openai_client(provider)
    memory_prompt = _load_text(SUBCATEGORY_MEMORY_PROMPT_PATH)
    revise_prompt = _load_text(REVISE_MEMORY_PROMPT_PATH)
    event_revision_suggestion_prompt = _load_text(
        EVENT_MEMORY_REVISION_SUGGESTION_PROMPT_PATH
    )
    batch_merge_prompt = _load_text(BATCH_MERGE_SUGGESTION_PROMPT_PATH)
    event_suggestion_semaphore = asyncio.Semaphore(20)

    revision_rows: list[dict[str, Any]] = []

    if (epoch == 1 and not force_revise) or not memory_output_path.exists():
        memory_entries: list[dict[str, Any]] = []
        grouped_items = sorted(current_groups.items())
        for (category_name, subcategory_name), rows in tqdm(
            grouped_items,
            desc=f"Epoch {epoch} memory init",
            unit="subcategory",
        ):
            for attempt in range(1, 4):
                try:
                    filtered_bundle = _build_trajectory_bundle(rows, sample_limit=sample_limit)
                    nofilter_bundle = _build_trajectory_bundle(
                        nofilter_groups.get((category_name, subcategory_name), []),
                        sample_limit=sample_limit,
                    )
                    user_prompt = memory_prompt.format(
                        subcategory_name=subcategory_name,
                        filtered_trajectory_bundle=_fmt_json(filtered_bundle),
                        nofilter_trajectory_bundle=_fmt_json(nofilter_bundle),
                        max_factors=max_factors,
                    )
                    parsed = await _chat_json(
                        client,
                        model=model_name,
                        system_prompt=(
                            "You synthesize reusable forecasting memory from grouped trajectories. "
                            "Return valid JSON only."
                        ),
                        user_prompt=user_prompt,
                        label=f"memory-init [{subcategory_name}]",
                    )
                    parsed["category_name"] = category_name
                    parsed["subcategory_name"] = subcategory_name
                    parsed["num_questions"] = len(rows)
                    parsed["num_sampled_questions"] = len(filtered_bundle)
                    memory_entries.append(parsed)
                    revision_rows.append(
                        {
                            "category_name": category_name,
                            "subcategory_name": subcategory_name,
                            "update_type": "initialize",
                            "epoch": epoch,
                            "source_results_path": str(current_results_path),
                            "nofilter_results_path": str(nofilter_results_path) if nofilter_results_path else None,
                            "generated_at": _now_iso(),
                            "initialized_memory": copy.deepcopy(parsed),
                        }
                    )
                    break
                except Exception as exc:
                    if attempt < 3:
                        wait = 2 ** attempt
                        print(
                            f"[memory-init] [{subcategory_name}] attempt {attempt} failed: {exc}. "
                            f"Retrying in {wait}s...",
                            flush=True,
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(
                            f"[memory-init] Skipping [{subcategory_name}] after 3 failed attempts: {exc}",
                            flush=True,
                        )

        memory_entries.sort(
            key=lambda entry: (str(entry.get("category_name", "")), str(entry.get("subcategory_name", "")))
        )
        clean_entries = _strip_memory_history(memory_entries)
        _save_json(memory_output_path, clean_entries)
        if snapshot_output_path != memory_output_path:
            _save_json(snapshot_output_path, clean_entries)
        _save_json(revision_report_output_path, revision_rows)
        return {
            "memory_output_path": str(memory_output_path),
            "snapshot_output_path": str(snapshot_output_path),
            "revision_report_output_path": str(revision_report_output_path),
            "num_entries": len(clean_entries),
            "skipped": False,
            "memory_entries": clean_entries,
            "revision_rows": revision_rows,
        }

    memory_entries = _load_memory_entries(memory_output_path)
    memory_lookup = _index_memory_entries(memory_entries)

    grouped_items = sorted(current_groups.items())
    subcategory_semaphore = asyncio.Semaphore(max(1, max_subcategory_concurrency))

    async def _process_subcategory(
        item: tuple[tuple[str, str], list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        (category_name, subcategory_name), current_group_rows = item
        key = (category_name, subcategory_name)
        async with subcategory_semaphore:
            existing_entry = copy.deepcopy(memory_lookup.get(key))

            if existing_entry is None:
                filtered_init_bundle = _build_trajectory_bundle(
                    current_group_rows,
                    sample_limit=sample_limit,
                )
                nofilter_init_bundle = _build_trajectory_bundle(
                    nofilter_groups.get(key, []),
                    sample_limit=sample_limit,
                )
                user_prompt = memory_prompt.format(
                    subcategory_name=subcategory_name,
                    filtered_trajectory_bundle=_fmt_json(filtered_init_bundle),
                    nofilter_trajectory_bundle=_fmt_json(nofilter_init_bundle),
                    max_factors=max_factors,
                )
                parsed = await _chat_json(
                    client,
                    model=model_name,
                    system_prompt=(
                        "You synthesize reusable forecasting memory from grouped trajectories. "
                        "Return valid JSON only."
                    ),
                    user_prompt=user_prompt,
                    label=f"memory-init-new [{subcategory_name}]",
                )
                parsed["category_name"] = category_name
                parsed["subcategory_name"] = subcategory_name
                parsed["num_questions"] = len(current_group_rows)
                parsed["num_sampled_questions"] = len(filtered_init_bundle)
                return parsed, {
                    "category_name": category_name,
                    "subcategory_name": subcategory_name,
                    "update_type": "initialize_missing_subcategory",
                    "epoch": epoch,
                    "source_results_path": str(current_results_path),
                    "nofilter_results_path": str(nofilter_results_path) if nofilter_results_path else None,
                    "generated_at": _now_iso(),
                    "initialized_memory": copy.deepcopy(parsed),
                }

            current_bundle = _build_trajectory_bundle(
                current_group_rows,
                sample_limit=sample_limit,
            )
            nofilter_rows_by_question = {
                str(row.get("question", "")): row
                for row in nofilter_groups.get(key, [])
            }
            compact_existing_entry = _compact_memory_entry_for_prompt(existing_entry)

            async def _get_one_event_suggestion(row: dict[str, Any]) -> dict[str, Any]:
                async with event_suggestion_semaphore:
                    filtered_event_bundle = _build_single_event_bundle(row)
                    nofilter_row = nofilter_rows_by_question.get(str(row.get("question", "")))
                    nofilter_event_bundle = (
                        _build_single_event_bundle(nofilter_row) if nofilter_row else {}
                    )
                    ground_truth = row.get("ground_truth")
                    prompt = event_revision_suggestion_prompt.format(
                        category_name=category_name,
                        subcategory_name=subcategory_name,
                        existing_memory_json=_fmt_json(compact_existing_entry),
                        filtered_event_bundle=_fmt_json(filtered_event_bundle),
                        nofilter_event_bundle=_fmt_json(nofilter_event_bundle),
                        ground_truth=_fmt_json(ground_truth),
                        max_factors=max_factors,
                    )
                    return await _chat_json(
                        client,
                        model=model_name,
                        system_prompt=(
                            "Suggest how one event should revise reusable subcategory forecasting "
                            "memory. Return valid JSON only."
                        ),
                        user_prompt=prompt,
                        label=f"event-suggest [{subcategory_name}]",
                    )

            event_revision_suggestions: list[dict[str, Any]] = list(
                await asyncio.gather(
                    *[_get_one_event_suggestion(row) for row in current_group_rows]
                )
            )

            # Merge suggestions in batches to keep the revision prompt tractable.
            # Skip batching when all suggestions fit in a single batch.
            if len(event_revision_suggestions) <= suggestion_batch_size:
                batch_summaries: list[dict[str, Any]] = event_revision_suggestions
            else:
                batches = [
                    event_revision_suggestions[i:i + suggestion_batch_size]
                    for i in range(0, len(event_revision_suggestions), suggestion_batch_size)
                ]
                batch_summaries = []
                for batch_idx, batch in enumerate(tqdm(
                    batches,
                    desc=f"Epoch {epoch} batch-merge {subcategory_name}",
                    unit="batch",
                    leave=False,
                )):
                    batch_summary = await _chat_json(
                        client,
                        model=model_name,
                        system_prompt=(
                            "Summarize recurring signals from a batch of per-event memory revision "
                            "suggestions. Return valid JSON only."
                        ),
                        user_prompt=batch_merge_prompt.format(
                            category_name=category_name,
                            subcategory_name=subcategory_name,
                            existing_memory_json=_fmt_json(compact_existing_entry),
                            event_suggestions_json=_fmt_json(batch),
                            batch_size=len(batch),
                            n_existing_questions=existing_entry.get("num_questions", 0),
                            max_factors=max_factors,
                        ),
                        label=f"batch-merge [{subcategory_name}] {batch_idx + 1}/{len(batches)}",
                    )
                    batch_summaries.append(batch_summary)

            user_prompt = revise_prompt.format(
                category_name=category_name,
                subcategory_name=subcategory_name,
                existing_memory_json=_fmt_json(compact_existing_entry),
                event_revision_suggestions_json=_fmt_json(batch_summaries),
                max_factors=max_factors,
                n_existing_questions=existing_entry.get("num_questions", 0),
                n_new_questions=len(current_group_rows),
            )
            revision = await _chat_json(
                client,
                model=model_name,
                system_prompt=(
                    "Revise reusable subcategory forecasting memory from event-level "
                    "revision suggestions. Return valid JSON only."
                ),
                user_prompt=user_prompt,
                label=f"memory-revise [{subcategory_name}]",
            )

            existing_entry["common_factors"] = revision.get(
                "revised_common_factors",
                existing_entry.get("common_factors", []),
            )
            existing_entry["common_reasoning_patterns"] = revision.get(
                "revised_common_reasoning_patterns",
                existing_entry.get(
                    "common_reasoning_patterns",
                    existing_entry.get("common_reasoning_procedure", []),
                ),
            )
            existing_entry["representative_examples"] = revision.get(
                "revised_representative_examples",
                existing_entry.get("representative_examples", []),
            )
            existing_entry["notes"] = revision.get("revised_notes", existing_entry.get("notes", ""))
            existing_entry["num_questions"] = len(current_group_rows)
            existing_entry["num_sampled_questions"] = len(current_bundle)

            revision_with_suggestions = copy.deepcopy(revision)
            revision_with_suggestions["event_revision_suggestions"] = event_revision_suggestions
            revision_with_suggestions["batch_summaries"] = batch_summaries
            revision_with_suggestions["category_name"] = category_name
            revision_with_suggestions["subcategory_name"] = subcategory_name
            revision_with_suggestions["update_type"] = "revise"
            revision_with_suggestions["epoch"] = epoch
            revision_with_suggestions["source_results_path"] = str(current_results_path)
            revision_with_suggestions["nofilter_results_path"] = (
                str(nofilter_results_path) if nofilter_results_path else None
            )
            revision_with_suggestions["generated_at"] = _now_iso()
            return existing_entry, revision_with_suggestions

    async def _process_subcategory_safe(
        item: tuple[tuple[str, str], list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        (category_name, subcategory_name), _ = item
        for attempt in range(1, 4):
            try:
                return await _process_subcategory(item)
            except Exception as exc:
                if attempt < 3:
                    wait = 2 ** attempt
                    print(
                        f"[memory-update] [{subcategory_name}] attempt {attempt} failed: {exc}. "
                        f"Retrying in {wait}s...",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                else:
                    print(
                        f"[memory-update] Skipping [{subcategory_name}] after 3 failed attempts: {exc}",
                        flush=True,
                    )
        return None

    pending_tasks = [
        asyncio.create_task(_process_subcategory_safe(item))
        for item in grouped_items
    ]

    updated_entries: list[dict[str, Any]] = []
    for completed_task in tqdm(
        asyncio.as_completed(pending_tasks),
        total=len(pending_tasks),
        desc=f"Epoch {epoch} memory update",
        unit="subcategory",
    ):
        result = await completed_task
        if result is None:
            continue
        updated_entry, revision_row = result
        updated_entries.append(updated_entry)
        if revision_row is not None:
            revision_rows.append(revision_row)

    updated_lookup = {
        (str(entry.get("category_name")), str(entry.get("subcategory_name"))): entry
        for entry in updated_entries
    }
    for index, candidate in enumerate(memory_entries):
        key = (str(candidate.get("category_name")), str(candidate.get("subcategory_name")))
        if key in updated_lookup:
            memory_entries[index] = updated_lookup[key]

    existing_keys = {
        (str(entry.get("category_name")), str(entry.get("subcategory_name")))
        for entry in memory_entries
    }
    for entry in updated_entries:
        key = (str(entry.get("category_name")), str(entry.get("subcategory_name")))
        if key not in existing_keys:
            memory_entries.append(entry)
            existing_keys.add(key)

    memory_entries.sort(
        key=lambda entry: (str(entry.get("category_name", "")), str(entry.get("subcategory_name", "")))
    )
    clean_entries = _strip_memory_history(memory_entries)
    _save_json(memory_output_path, clean_entries)
    if snapshot_output_path != memory_output_path:
        _save_json(snapshot_output_path, clean_entries)
    _save_json(revision_report_output_path, revision_rows)

    return {
        "memory_output_path": str(memory_output_path),
        "snapshot_output_path": str(snapshot_output_path),
        "revision_report_output_path": str(revision_report_output_path),
        "num_entries": len(clean_entries),
        "skipped": False,
        "memory_entries": clean_entries,
        "revision_rows": revision_rows,
    }


# Run forecasting with optional subcategory memory injection.

async def inference(
    *,
    selected_week_csv: str,
    week_index: int,
    output_dir: str | Path,
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 20,
    search_provider: str = "serper",
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
    classification_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    use_memory: bool = True,
    use_factor_memory_component: bool = True,
    use_reasoning_memory_component: bool = True,
    factor_memory_path: str | Path | None = None,
    first_k: int | None = None,
    max_concurrency: int = 5,
    results_filename: str | None = None,
    evaluation_filename: str = "evaluation.json",
    resume_results_path: str | Path | None = None,
    provider: str = "azure",
) -> dict[str, Any]:
    client = _make_openai_client(provider)
    set_default_openai_client(client, use_for_tracing=False)
    set_tracing_disabled(True)
    max_turns = max_search_calls
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_rows = _load_json(classification_path) if classification_path else []
    classification_lookup = _classification_lookup(classification_rows)
    memory_entries = _load_memory_entries(memory_path) if memory_path and Path(memory_path).exists() else []
    memory_lookup = _index_memory_entries(memory_entries)

    factor_memory = None
    use_factor_memory = False
    if factor_memory_path and Path(factor_memory_path).exists():
        factor_memory = load_factor_memory(factor_memory_path)
        use_factor_memory = True

    weekly_tasks = load_weekly_tasks(selected_week_csv, first_k=first_k)
    results_by_idx: dict[int, dict[str, Any]] = {}
    start_task_idx = 0
    pending_rerun_indices: list[int] = []

    if resume_results_path is not None and Path(resume_results_path).exists():
        saved_payload = load_results_payload(resume_results_path)
        saved_samples = saved_payload.get("samples", [])
        start_task_idx = len(saved_samples)
        pending_rerun_indices = [
            idx for idx, s in enumerate(saved_samples)
            if s.get("error") or str(s.get("status") or "").strip().lower() != "success"
        ]
        for idx, s in enumerate(saved_samples):
            if idx not in pending_rerun_indices:
                results_by_idx[idx] = s
        print(
            f"Resuming: loaded {len(results_by_idx)} completed tasks from {resume_results_path} "
            f"({len(pending_rerun_indices)} needing rerun)"
        )

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(task_idx: int, sample_task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _, formatted_question = create_prediction_prompt(sample_task)

        classification = classification_lookup.get(formatted_question, {})
        if not classification:
            classification = classification_lookup.get(
                _normalize_question_text(formatted_question),
                {},
            )
        category_name = classification.get("category_name")
        subcategory_name = classification.get("subcategory_name")
        memory_entry = None
        external_memory_factor = None
        external_memory_factor_raw = None
        external_memory_reasoning = None
        external_memory_reasoning_raw = None
        external_memory_metadata: dict[str, Any] = {}

        if use_memory and category_name and subcategory_name:
            memory_entry = memory_lookup.get((str(category_name), str(subcategory_name)))
            if memory_entry is not None:
                if use_factor_memory_component:
                    external_memory_factor = build_factor_memory_context_from_subcategory(memory_entry)
                    external_memory_factor_raw = copy.deepcopy(
                        {
                            "category_name": memory_entry.get("category_name"),
                            "subcategory_name": memory_entry.get("subcategory_name"),
                            "common_factors": memory_entry.get("common_factors", []),
                        }
                    )
                if use_reasoning_memory_component:
                    external_memory_reasoning = build_reasoning_memory_context_from_subcategory(memory_entry)
                    external_memory_reasoning_raw = copy.deepcopy(
                        {
                            "category_name": memory_entry.get("category_name"),
                            "subcategory_name": memory_entry.get("subcategory_name"),
                            "common_reasoning_patterns": memory_entry.get(
                                "common_reasoning_patterns",
                                memory_entry.get("common_reasoning_procedure", []),
                            ),
                            "representative_examples": memory_entry.get("representative_examples", []),
                            "notes": memory_entry.get("notes", ""),
                        }
                    )
                external_memory_metadata = {
                    "category_name": category_name,
                    "subcategory_name": subcategory_name,
                }

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
                        factor_memory=factor_memory,
                        use_factor_memory=use_factor_memory,
                        external_memory_factor=external_memory_factor,
                        external_memory_factor_raw=external_memory_factor_raw,
                        external_memory_reasoning=external_memory_reasoning,
                        external_memory_reasoning_raw=external_memory_reasoning_raw,
                        external_memory_metadata=external_memory_metadata,
                    )
                    break
                except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                    if attempt == max_retries - 1:
                        raise
                    wait = min(2 ** attempt * 10, 180)
                    print(f"  [task {task_idx}] rate limit, retrying in {wait}s ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
        return task_idx, prediction_result["data"]

    task_indices_to_run = pending_rerun_indices + list(range(start_task_idx, len(weekly_tasks)))
    pending_tasks = [
        asyncio.create_task(_run_one(task_idx, weekly_tasks[task_idx]))
        for task_idx in task_indices_to_run
    ]

    n_total = len(pending_tasks)
    n_done = 0
    for completed_task in tqdm(
        asyncio.as_completed(pending_tasks),
        total=n_total,
        desc="Inference",
        unit="event",
    ):
        task_idx, sample_result = await completed_task
        n_done += 1
        print(f"[inference] completed {n_done}/{n_total} (task_idx={task_idx})", flush=True)
        results_by_idx[task_idx] = sample_result
        sample_results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

        results_payload = build_week_results_payload(
            selected_week_csv=selected_week_csv,
            week_idx=week_index,
            model_name=model_name,
            max_search_calls=max_search_calls,
            search_provider=search_provider,
            filter_year=filter_year,
            filter_days_before_close=filter_days_before_close,
            use_close_date_filter=use_close_date_filter,
            sample_results=sample_results,
        )
        if results_filename:
            _checkpoint_path = output_dir / results_filename
            with _checkpoint_path.open("w", encoding="utf-8") as _f:
                metadata = {k: v for k, v in results_payload.items() if k != "samples"}
                _f.write(json.dumps({"record_type": "metadata", **metadata}, ensure_ascii=True) + "\n")
                for _s in results_payload.get("samples", []):
                    _f.write(json.dumps({"record_type": "sample", **_s}, ensure_ascii=True) + "\n")
        else:
            save_week_results(
                results_payload,
                output_dir,
                use_close_date_filter=use_close_date_filter,
            )

    sample_results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

    if results_filename:
        results_path = output_dir / results_filename
    else:
        results_path = resolve_results_path(output_dir, use_close_date_filter)
    evaluation_payload = evaluate_predictions(
        results_path, save=True, output_filename=evaluation_filename
    )
    return {
        "results_path": str(results_path),
        "evaluation_path": str(output_dir / evaluation_filename),
        "num_samples": len(sample_results),
        "evaluation": evaluation_payload,
    }


# Orchestrate the full multi-epoch memory pipeline.

async def run_epoch_pipeline(
    *,
    baseline_results_path: str | Path,
    epochs: int = 4,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    dataset: str = "prophet_arena",
    model_name: str = "gpt-5-mini",
    max_search_calls: int = 20,
    search_provider: str = "serper",
    filter_year: int | None = None,
    filter_days_before_close: int = 2,
    use_close_date_filter: bool = True,
    factor_memory_path: str | Path | None = None,
    first_k: int | None = None,
    max_concurrency: int = 5,
    max_factors: int = 8,
    suggestion_batch_size: int = 30,
    overwrite_preprocess: bool = False,
    overwrite_epoch1_memory: bool = False,
    start_epoch: int = 1,
    initial_memory_path: str | Path | None = None,
    update_classification: bool = False,
    provider: str = "azure",
) -> dict[str, Any]:
    reset_cost_tracker()
    baseline_results_path = resolve_existing_results_input_path(baseline_results_path, prefer_filter=True)
    baseline_nofilter_path = resolve_existing_results_input_path(
        baseline_results_path.parent, prefer_filter=False
    )
    if baseline_nofilter_path == baseline_results_path:
        baseline_nofilter_path = None
    # Root all outputs under the week folder, not the baselines subfolder.
    _results_parent = baseline_results_path.parent
    base_dir = _results_parent.parent if _results_parent.name == "baselines" else _results_parent
    epochs_dir = base_dir / "memory_epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    reset_cost_tracker()

    # If the caller didn't specify a taxonomy path, prefer the locally updated taxonomy
    # saved by a previous run of this pipeline over the global init_ctgr default.
    # This ensures taxonomy expansions from earlier runs (or earlier weeks when the
    # caller points taxonomy_path to last week's file) are not silently discarded.
    local_taxonomy_file = base_dir / f"{dataset}_ctgr.json"
    effective_taxonomy_path = Path(taxonomy_path)
    if effective_taxonomy_path == DEFAULT_TAXONOMY_PATH and local_taxonomy_file.exists():
        effective_taxonomy_path = local_taxonomy_file
        print(f"Using locally updated taxonomy: {effective_taxonomy_path}", flush=True)

    preprocess_outputs = await preprocess(
        results_path=baseline_results_path,
        taxonomy_path=effective_taxonomy_path,
        classification_output_path=base_dir / "question_category_classification.json",
        grouped_proposals_output_path=base_dir / "proposed_new_category_groups.json",
        verified_proposals_output_path=base_dir / "verified_new_category_proposals.json",
        taxonomy_backup_path=base_dir / f"{dataset}_ctgr_before_merge.json",
        taxonomy_output_path=base_dir / f"{dataset}_ctgr.json",
        model_name=model_name,
        overwrite=overwrite_preprocess,
        skip_classification=not update_classification,
        provider=provider,
    )
    classification_path = Path(preprocess_outputs["classification_output_path"])

    latest_memory_path = base_dir / "memory.json"
    pipeline_log_path = epochs_dir / "pipeline_log.jsonl"
    epoch1_dir = epochs_dir / "epoch_1"
    epoch1_dir.mkdir(parents=True, exist_ok=True)

    # Seed from an existing memory so epoch 1 revises rather than initialises from scratch.
    if initial_memory_path is not None and not latest_memory_path.exists():
        import shutil
        shutil.copy2(Path(initial_memory_path), latest_memory_path)
        print(f"Seeded memory from {initial_memory_path}", flush=True)
    baseline_evaluation_path = base_dir / "evaluation.json"
    epoch1_training_source = (
        baseline_evaluation_path if baseline_evaluation_path.exists() else baseline_results_path
    )

    baseline_payload = load_results_payload(baseline_results_path)
    selected_week_csv = str(baseline_payload["selected_week_csv"])
    week_index = int(baseline_payload.get("week_index", 0))

    if start_epoch <= 1:
        epoch1_train = await train(
            current_results_path=epoch1_training_source,
            classification_path=classification_path,
            memory_output_path=latest_memory_path,
            snapshot_output_path=epoch1_dir / "memory.json",
            revision_report_output_path=epoch1_dir / "memory_revision_epoch_1.json",
            epoch=1,
            model_name=model_name,
            nofilter_results_path=baseline_nofilter_path,
            max_factors=max_factors,
            suggestion_batch_size=suggestion_batch_size,
            overwrite=overwrite_epoch1_memory,
            force_revise=initial_memory_path is not None,
            provider=provider,
        )

        # Log epoch 1 memory init records.
        _append_jsonl(
            pipeline_log_path,
            [
                {
                    "type": "memory_init",
                    "epoch": 1,
                    "category_name": entry.get("category_name"),
                    "subcategory_name": entry.get("subcategory_name"),
                    "memory": entry,
                    "generated_at": _now_iso(),
                }
                for entry in epoch1_train["memory_entries"]
            ],
        )

        epoch_runs: list[dict[str, Any]] = [
            {
                "epoch": 1,
                "results_path": str(baseline_results_path),
                "evaluation_path": str(base_dir / "evaluation.json"),
                "memory_path": str(latest_memory_path),
                "memory_snapshot_path": str(epoch1_dir / "memory.json"),
                "train": {k: v for k, v in epoch1_train.items() if k not in ("memory_entries", "revision_rows")},
            }
        ]
    else:
        print(f"Resuming from epoch {start_epoch} — skipping epoch 1 memory init.")
        if not latest_memory_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: epoch-1 memory not found at {latest_memory_path}. "
                "Run without --start-epoch first to build it."
            )
        # Restore prior epoch_runs from saved pipeline summary if available.
        summary_path = epochs_dir / "pipeline_summary.json"
        if summary_path.exists():
            prior_summary = _load_json(summary_path)
            epoch_runs = [r for r in prior_summary.get("epochs", []) if r["epoch"] < start_epoch]
        else:
            epoch_runs = []

    for epoch in tqdm(range(max(2, start_epoch), epochs + 1), desc="Epochs", unit="epoch"):
        epoch_dir = epochs_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        is_last_epoch = epoch == epochs

        _existing_results = resolve_results_path(epoch_dir, use_close_date_filter)
        _existing_evaluation = epoch_dir / "evaluation.json"
        if _existing_results.exists() and _existing_evaluation.exists():
            print(
                f"Epoch {epoch}: inference results already exist at {_existing_results}, skipping inference.",
                flush=True,
            )
            _eval_payload = _load_json(_existing_evaluation)
            inference_outputs = {
                "results_path": str(_existing_results),
                "evaluation_path": str(_existing_evaluation),
                "num_samples": len((_eval_payload.get("weeks") or [{}])[0].get("samples", [])),
                "evaluation": _eval_payload,
            }
        else:
            _resume_path = _existing_results if _existing_results.exists() else None
            if _resume_path:
                print(
                    f"Epoch {epoch}: resuming partial inference from {_resume_path}.",
                    flush=True,
                )
            inference_outputs = await inference(
                selected_week_csv=selected_week_csv,
                week_index=week_index,
                output_dir=epoch_dir,
                model_name=model_name,
                max_search_calls=max_search_calls,
                search_provider=search_provider,
                filter_year=filter_year,
                filter_days_before_close=filter_days_before_close,
                use_close_date_filter=use_close_date_filter,
                classification_path=classification_path,
                memory_path=latest_memory_path,
                use_memory=True,
                factor_memory_path=factor_memory_path,
                first_k=first_k,
                max_concurrency=max_concurrency,
                resume_results_path=_resume_path,
                provider=provider,
            )

        log_records: list[dict[str, Any]] = []

        # Per-sample inference records with Brier score.
        evaluation = inference_outputs.get("evaluation") or {}
        weeks = evaluation.get("weeks") or []
        epoch_samples = weeks[0].get("samples", []) if weeks else []
        for sample in epoch_samples:
            log_records.append(
                {
                    "type": "inference_sample",
                    "epoch": epoch,
                    "question": sample.get("question"),
                    "prediction": sample.get("predictions"),
                    "ground_truth": sample.get("outcomes"),
                    "brier_score": sample.get("brier_score"),
                    "generated_at": _now_iso(),
                }
            )
        log_records.append(
            {
                "type": "epoch_brier",
                "epoch": epoch,
                "average_brier_score": evaluation.get("average_brier_score"),
                "generated_at": _now_iso(),
            }
        )

        epoch_run: dict[str, Any] = {
            "epoch": epoch,
            "results_path": inference_outputs["results_path"],
            "evaluation_path": inference_outputs["evaluation_path"],
            "memory_path": str(latest_memory_path),
            "memory_snapshot_path": str(epoch_dir / "memory.json"),
            "inference": {k: v for k, v in inference_outputs.items() if k != "evaluation"},
        }

        if not is_last_epoch:
            train_outputs = await train(
                current_results_path=inference_outputs["evaluation_path"],
                nofilter_results_path=baseline_nofilter_path,
                classification_path=classification_path,
                memory_output_path=latest_memory_path,
                snapshot_output_path=epoch_dir / "memory.json",
                revision_report_output_path=epoch_dir / f"memory_revision_epoch_{epoch}.json",
                epoch=epoch,
                model_name=model_name,
                max_factors=max_factors,
                suggestion_batch_size=suggestion_batch_size,
                provider=provider,
            )

            for revision_row in train_outputs["revision_rows"]:
                for suggestion in revision_row.get("event_revision_suggestions", []):
                    log_records.append(
                        {
                            "type": "event_suggestion",
                            "epoch": epoch,
                            "suggestion": suggestion,
                            "generated_at": _now_iso(),
                        }
                    )
                merged = {k: v for k, v in revision_row.items() if k != "event_revision_suggestions"}
                log_records.append(
                    {
                        "type": "merged_revision",
                        "epoch": epoch,
                        "revision": merged,
                        "generated_at": _now_iso(),
                    }
                )

            for entry in train_outputs["memory_entries"]:
                log_records.append(
                    {
                        "type": "memory_update",
                        "epoch": epoch,
                        "category_name": entry.get("category_name"),
                        "subcategory_name": entry.get("subcategory_name"),
                        "memory": entry,
                        "generated_at": _now_iso(),
                    }
                )

            epoch_run["train"] = {k: v for k, v in train_outputs.items() if k not in ("memory_entries", "revision_rows")}

        _append_jsonl(pipeline_log_path, log_records)
        epoch_runs.append(epoch_run)

    pipeline_summary = {
        "baseline_results_path": str(baseline_results_path),
        "classification_path": str(classification_path),
        "taxonomy_path": str(taxonomy_path),
        "latest_memory_path": str(latest_memory_path),
        "pipeline_log_path": str(pipeline_log_path),
        "cost_summary": get_cost_summary(),
        "epochs": epoch_runs,
        "generated_at": _now_iso(),
    }
    _save_json(epochs_dir / "pipeline_summary.json", pipeline_summary)
    _save_json(epochs_dir / "cost_summary.json", pipeline_summary["cost_summary"])
    return pipeline_summary


# Compare baseline evaluated data against a with-memory rerun.

def _derive_memory_effectiveness_dir(
    memory_path: Path,
    evaluated_data_path: Path,
    output_dir: str | Path | None,
) -> Path:
    import re as _re
    _mem_parts = memory_path.parts
    _week_part = next((p for p in reversed(_mem_parts) if p.lower().startswith("week")), None)
    _epoch_part = next((p for p in reversed(_mem_parts) if p.lower().startswith("epoch")), None)
    if _week_part and _epoch_part:
        _memory_label = f"{_week_part}_{_epoch_part}"
    elif _week_part:
        _memory_label = _week_part
    else:
        _memory_label = memory_path.parent.name

    is_csv = evaluated_data_path.suffix == ".csv" and evaluated_data_path.is_file()
    if is_csv:
        # Derive output under results/ using model/search from memory_path and week from CSV name.
        csv_week_match = _re.search(r"(week\d+)", evaluated_data_path.stem, _re.IGNORECASE)
        csv_week = csv_week_match.group(1).lower() if csv_week_match else "week0"
        csv_dataset = evaluated_data_path.parent.name  # e.g. "futurex_online"
        try:
            results_idx = next(i for i, p in enumerate(_mem_parts) if p == "results")
            model = _mem_parts[results_idx + 2]
            search = _mem_parts[results_idx + 3]
        except (StopIteration, IndexError):
            model, search = "unknown", "unknown"
        _output_base = Path("results") / csv_dataset / model / search / csv_week
    else:
        _output_base = evaluated_data_path if evaluated_data_path.is_dir() else evaluated_data_path.parent
        if _output_base.name == "baselines":
            _output_base = _output_base.parent

    return Path(output_dir or _output_base / "memory_effectiveness" / _memory_label)


async def test_memory_effectiveness(
    *,
    memory_path: str | Path,
    evaluated_data_path: str | Path,
    classification_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    output_dir: str | Path | None = None,
    final_output_path: str | Path | None = None,
    model_name: str | None = None,
    max_search_calls: int | None = None,
    search_provider: str | None = None,
    filter_year: int | None = None,
    filter_days_before_close: int | None = None,
    use_close_date_filter: bool | None = None,
    factor_memory_path: str | Path | None = None,
    first_k: int | None = None,
    max_concurrency: int = 5,
    overwrite_preprocess: bool = False,
    skip_classification: bool = False,
    provider: str = "azure",
    resume_results_path: str | Path | None = None,
) -> dict[str, Any]:
    memory_path = Path(memory_path)
    evaluated_data_path = Path(evaluated_data_path)
    print(f"test_memory_effectiveness: memory={memory_path.name}, data={evaluated_data_path}", flush=True)
    prefer_filter = False if use_close_date_filter is False else True

    # CSV path: run inference directly without a pre-existing baseline.
    csv_only_mode = evaluated_data_path.suffix == ".csv" and evaluated_data_path.is_file()

    if csv_only_mode:
        selected_week_csv = str(evaluated_data_path)
        match = __import__("re").search(r"week(\d+)", evaluated_data_path.stem, __import__("re").IGNORECASE)
        week_index = int(match.group(1)) - 1 if match else 0
        baseline_results_path = None
        baseline_evaluation_payload = {}
        pass
    elif evaluated_data_path.is_dir():
        baseline_results_path = resolve_existing_results_input_path(
            evaluated_data_path / "baselines",
            prefer_filter=prefer_filter,
        )
        if not baseline_results_path.exists():
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path,
                prefer_filter=prefer_filter,
            )
        baseline_evaluation_path = _resolve_baseline_evaluation_path(
            evaluated_data_path,
            prefer_filter=prefer_filter,
        )
        if baseline_evaluation_path is not None:
            print(f"Using baseline evaluation from {baseline_evaluation_path}", flush=True)
            baseline_evaluation_payload = _load_json(baseline_evaluation_path)
        else:
            baseline_evaluation_payload = evaluate_predictions(baseline_results_path, save=False)
        pass
    else:
        evaluated_payload = _load_json(evaluated_data_path)
        if "average_brier_score" in evaluated_payload:
            baseline_evaluation_payload = evaluated_payload
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path.parent / "baselines", prefer_filter=prefer_filter,
            )
            if not baseline_results_path.exists():
                baseline_results_path = resolve_existing_results_input_path(
                    evaluated_data_path.parent,
                    prefer_filter=prefer_filter,
                )
        else:
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path, prefer_filter=prefer_filter,
            )
            baseline_evaluation_payload = evaluate_predictions(baseline_results_path, save=False)
        pass

    if not csv_only_mode and not baseline_results_path.exists():
        raise ValueError(
            f"Could not find a saved results file next to {evaluated_data_path}."
        )

    if not csv_only_mode:
        baseline_results_payload = load_results_payload(baseline_results_path)
        selected_week_csv = str(baseline_results_payload["selected_week_csv"])
        week_index = int(baseline_results_payload.get("week_index", 0))
    else:
        baseline_results_payload = {}

    output_dir = _derive_memory_effectiveness_dir(memory_path, evaluated_data_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_path_default = output_dir / "question_category_classification.json"
    if skip_classification:
        resolved_classification_path = Path(classification_path or classification_path_default)
        if not resolved_classification_path.exists():
            raise FileNotFoundError(
                f"--skip-classification requires an existing classification file at "
                f"{resolved_classification_path}, but none was found."
            )
        print(f"Skipping classification — loading from {resolved_classification_path}", flush=True)
        classification_path = resolved_classification_path
    else:
        classification_path = await _ensure_classification_path(
            classification_path=classification_path or classification_path_default,
            taxonomy_path=taxonomy_path,
            baseline_results_path=baseline_results_path,
            csv_only_mode=csv_only_mode,
            model_name=model_name,
            overwrite_preprocess=overwrite_preprocess,
            output_dir=output_dir,
            provider=provider,
        )
        # csv_only_mode: no baseline results to classify from, so classify questions
        # directly from the weekly CSV.
        if csv_only_mode and not classification_path.exists():
            tasks = load_weekly_tasks(selected_week_csv)
            questions = [t.get("question") or t.get("title") or "" for t in tasks if t.get("question") or t.get("title")]
            classification_path = await _classify_questions(
                questions=questions,
                taxonomy_path=taxonomy_path,
                classification_output_path=classification_path,
                model_name=model_name or "gpt-5-mini",
                provider=provider,
            )

    effective_model_name = model_name or str(
        baseline_results_payload.get("model") or baseline_evaluation_payload.get("model") or "gpt-5-mini"
    )
    effective_max_search_calls = int(
        max_search_calls if max_search_calls is not None
        else baseline_results_payload.get("max_search_calls", 20)
    )
    effective_search_provider = str(
        search_provider
        or baseline_results_payload.get("search_provider")
        or baseline_evaluation_payload.get("search_provider")
        or "serper"
    )
    effective_filter_year = (
        filter_year if filter_year is not None else baseline_results_payload.get("filter_year")
    )
    effective_filter_days_before_close = int(
        filter_days_before_close
        if filter_days_before_close is not None
        else baseline_results_payload.get("filter_days_before_close", 2)
    )
    effective_use_close_date_filter = bool(
        use_close_date_filter
        if use_close_date_filter is not None
        else baseline_results_payload.get("use_close_date_filter", True)
    )

    with_memory_outputs = await inference(
        selected_week_csv=selected_week_csv,
        week_index=week_index,
        output_dir=output_dir,
        model_name=effective_model_name,
        max_search_calls=effective_max_search_calls,
        search_provider=effective_search_provider,
        filter_year=effective_filter_year,
        filter_days_before_close=effective_filter_days_before_close,
        use_close_date_filter=effective_use_close_date_filter,
        classification_path=classification_path,
        memory_path=memory_path,
        use_memory=True,
        factor_memory_path=factor_memory_path,
        first_k=first_k,
        max_concurrency=max_concurrency,
        resume_results_path=resume_results_path,
        provider=provider,
    )

    baseline_brier = baseline_evaluation_payload.get("average_brier_score")
    with_memory_brier = with_memory_outputs["evaluation"].get("average_brier_score")
    delta_brier = None
    if baseline_brier is not None and with_memory_brier is not None:
        delta_brier = with_memory_brier - baseline_brier

    baseline_ece = baseline_evaluation_payload.get("ece")
    with_memory_ece = with_memory_outputs["evaluation"].get("ece")
    delta_ece = None
    if baseline_ece is not None and with_memory_ece is not None:
        delta_ece = with_memory_ece - baseline_ece

    comparison_summary = {
        "baseline_evaluated_data_path": str(evaluated_data_path),
        "baseline_results_path": str(baseline_results_path) if baseline_results_path else None,
        "classification_path": str(classification_path),
        "memory_path": str(memory_path),
        "with_memory_results_path": with_memory_outputs["results_path"],
        "with_memory_evaluation_path": with_memory_outputs["evaluation_path"],
        "baseline_average_brier_score": baseline_brier,
        "with_memory_average_brier_score": with_memory_brier,
        "delta_brier_score": delta_brier,
        "baseline_ece": baseline_ece,
        "with_memory_ece": with_memory_ece,
        "delta_ece": delta_ece,
        "max_turns": effective_max_search_calls,
        "use_close_date_filter": effective_use_close_date_filter,
        "improved": (delta_brier is not None and delta_brier < 0),
        "generated_at": _now_iso(),
    }
    summary_output_path = Path(final_output_path) if final_output_path else output_dir / "effectiveness_summary.json"
    _save_json(summary_output_path, comparison_summary)
    comparison_summary["summary_output_path"] = str(summary_output_path)
    return comparison_summary


async def run_memory_ablations(
    *,
    memory_path: str | Path,
    evaluated_data_path: str | Path,
    classification_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    max_search_calls: int | None = None,
    search_provider: str | None = None,
    filter_year: int | None = None,
    filter_days_before_close: int | None = None,
    use_close_date_filter: bool | None = None,
    factor_memory_path: str | Path | None = None,
    first_k: int | None = None,
    max_concurrency: int = 5,
    overwrite_preprocess: bool = False,
    skip_classification: bool = False,
    provider: str = "azure",
    resume_results_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ablation experiments (no factor memory, no reasoning memory) without
    re-running the full-memory inference. Results are saved under
    week$N/memory_effectiveness/$memory_label/ablation_*/."""
    memory_path = Path(memory_path)
    evaluated_data_path = Path(evaluated_data_path)
    print(f"run_memory_ablations: memory={memory_path.name}, data={evaluated_data_path}", flush=True)
    prefer_filter = False if use_close_date_filter is False else True

    csv_only_mode = evaluated_data_path.suffix == ".csv" and evaluated_data_path.is_file()

    if csv_only_mode:
        selected_week_csv = str(evaluated_data_path)
        match = __import__("re").search(r"week(\d+)", evaluated_data_path.stem, __import__("re").IGNORECASE)
        week_index = int(match.group(1)) - 1 if match else 0
        baseline_results_path = None
        baseline_evaluation_payload = {}
        pass
    elif evaluated_data_path.is_dir():
        baseline_results_path = resolve_existing_results_input_path(
            evaluated_data_path / "baselines",
            prefer_filter=prefer_filter,
        )
        if not baseline_results_path.exists():
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path,
                prefer_filter=prefer_filter,
            )
        baseline_evaluation_path = _resolve_baseline_evaluation_path(
            evaluated_data_path,
            prefer_filter=prefer_filter,
        )
        if baseline_evaluation_path is not None:
            baseline_evaluation_payload = _load_json(baseline_evaluation_path)
        else:
            baseline_evaluation_payload = evaluate_predictions(baseline_results_path, save=False)
        pass
    else:
        evaluated_payload = _load_json(evaluated_data_path)
        if "average_brier_score" in evaluated_payload:
            baseline_evaluation_payload = evaluated_payload
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path.parent / "baselines", prefer_filter=prefer_filter,
            )
            if not baseline_results_path.exists():
                baseline_results_path = resolve_existing_results_input_path(
                    evaluated_data_path.parent,
                    prefer_filter=prefer_filter,
                )
        else:
            baseline_results_path = resolve_existing_results_input_path(
                evaluated_data_path, prefer_filter=prefer_filter,
            )
            baseline_evaluation_payload = evaluate_predictions(baseline_results_path, save=False)
        pass

    if not csv_only_mode and not baseline_results_path.exists():
        raise ValueError(
            f"Could not find a saved results file next to {evaluated_data_path}."
        )

    if not csv_only_mode:
        baseline_results_payload = load_results_payload(baseline_results_path)
        selected_week_csv = str(baseline_results_payload["selected_week_csv"])
        week_index = int(baseline_results_payload.get("week_index", 0))
    else:
        baseline_results_payload = {}

    base_output_dir = _derive_memory_effectiveness_dir(memory_path, evaluated_data_path, output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    classification_path_default = base_output_dir / "question_category_classification.json"
    if skip_classification:
        resolved_classification_path = Path(classification_path or classification_path_default)
        if not resolved_classification_path.exists():
            raise FileNotFoundError(
                f"--skip-classification requires an existing classification file at "
                f"{resolved_classification_path}, but none was found."
            )
        print(f"Skipping classification — loading from {resolved_classification_path}", flush=True)
        classification_path = resolved_classification_path
    else:
        classification_path = await _ensure_classification_path(
            classification_path=classification_path or classification_path_default,
            taxonomy_path=taxonomy_path,
            baseline_results_path=baseline_results_path,
            csv_only_mode=csv_only_mode,
            model_name=model_name,
            overwrite_preprocess=overwrite_preprocess,
            output_dir=base_output_dir,
            provider=provider,
        )

    effective_model_name = model_name or str(
        baseline_results_payload.get("model") or baseline_evaluation_payload.get("model") or "gpt-5-mini"
    )
    effective_max_search_calls = int(
        max_search_calls if max_search_calls is not None
        else baseline_results_payload.get("max_search_calls", 20)
    )
    effective_search_provider = str(
        search_provider
        or baseline_results_payload.get("search_provider")
        or baseline_evaluation_payload.get("search_provider")
        or "serper"
    )
    effective_filter_year = (
        filter_year if filter_year is not None else baseline_results_payload.get("filter_year")
    )
    effective_filter_days_before_close = int(
        filter_days_before_close
        if filter_days_before_close is not None
        else baseline_results_payload.get("filter_days_before_close", 2)
    )
    effective_use_close_date_filter = bool(
        use_close_date_filter
        if use_close_date_filter is not None
        else baseline_results_payload.get("use_close_date_filter", True)
    )

    _shared_kwargs = dict(
        selected_week_csv=selected_week_csv,
        week_index=week_index,
        model_name=effective_model_name,
        max_search_calls=effective_max_search_calls,
        search_provider=effective_search_provider,
        filter_year=effective_filter_year,
        filter_days_before_close=effective_filter_days_before_close,
        use_close_date_filter=effective_use_close_date_filter,
        classification_path=classification_path,
        memory_path=memory_path,
        use_memory=True,
        factor_memory_path=factor_memory_path,
        first_k=first_k,
        max_concurrency=max_concurrency,
        provider=provider,
    )

    _filter_suffix = "with_filter" if effective_use_close_date_filter else "no_filter"
    _no_factor_file = f"no_factor_results_{_filter_suffix}.jsonl"
    _no_reasoning_file = f"no_reasoning_results_{_filter_suffix}.jsonl"

    _no_factor_resume = base_output_dir / _no_factor_file
    _no_reasoning_resume = base_output_dir / _no_reasoning_file

    print("\n[Ablation] Running without factor memory component...", flush=True)
    no_factor_outputs = await inference(
        output_dir=base_output_dir,
        use_factor_memory_component=False,
        use_reasoning_memory_component=True,
        results_filename=_no_factor_file,
        evaluation_filename="no_factor_evaluation.json",
        resume_results_path=_no_factor_resume if _no_factor_resume.exists() else None,
        **_shared_kwargs,
    )

    print("\n[Ablation] Running without reasoning memory component...", flush=True)
    no_reasoning_outputs = await inference(
        output_dir=base_output_dir,
        use_factor_memory_component=True,
        use_reasoning_memory_component=False,
        results_filename=_no_reasoning_file,
        evaluation_filename="no_reasoning_evaluation.json",
        resume_results_path=_no_reasoning_resume if _no_reasoning_resume.exists() else None,
        **_shared_kwargs,
    )

    def _brier_ece(outputs: dict[str, Any]) -> tuple[float | None, float | None]:
        ev = outputs.get("evaluation") or {}
        return ev.get("average_brier_score"), ev.get("ece")

    def _delta(a: float | None, b: float | None) -> float | None:
        return (b - a) if (a is not None and b is not None) else None

    baseline_brier = baseline_evaluation_payload.get("average_brier_score")
    baseline_ece = baseline_evaluation_payload.get("ece")
    no_factor_brier, no_factor_ece = _brier_ece(no_factor_outputs)
    no_reasoning_brier, no_reasoning_ece = _brier_ece(no_reasoning_outputs)

    # Load existing full-memory effectiveness summary if available (for delta_vs_full_memory).
    _existing_summary_path = base_output_dir / "effectiveness_summary.json"
    with_memory_brier: float | None = None
    with_memory_ece: float | None = None
    if _existing_summary_path.exists():
        try:
            _existing = _load_json(_existing_summary_path)
            with_memory_brier = _existing.get("with_memory_average_brier_score")
            with_memory_ece = _existing.get("with_memory_ece")
        except Exception:
            pass

    ablation_summary: dict[str, Any] = {
        "baseline_evaluated_data_path": str(evaluated_data_path),
        "baseline_results_path": str(baseline_results_path) if baseline_results_path else None,
        "classification_path": str(classification_path),
        "memory_path": str(memory_path),
        "baseline_average_brier_score": baseline_brier,
        "baseline_ece": baseline_ece,
        "with_memory_average_brier_score": with_memory_brier,
        "with_memory_ece": with_memory_ece,
        "ablations": {
            "no_factor_memory": {
                "results_path": no_factor_outputs["results_path"],
                "evaluation_path": no_factor_outputs["evaluation_path"],
                "average_brier_score": no_factor_brier,
                "ece": no_factor_ece,
                "delta_brier_vs_baseline": _delta(baseline_brier, no_factor_brier),
                "delta_brier_vs_full_memory": _delta(with_memory_brier, no_factor_brier),
                "delta_ece_vs_baseline": _delta(baseline_ece, no_factor_ece),
                "delta_ece_vs_full_memory": _delta(with_memory_ece, no_factor_ece),
            },
            "no_reasoning_memory": {
                "results_path": no_reasoning_outputs["results_path"],
                "evaluation_path": no_reasoning_outputs["evaluation_path"],
                "average_brier_score": no_reasoning_brier,
                "ece": no_reasoning_ece,
                "delta_brier_vs_baseline": _delta(baseline_brier, no_reasoning_brier),
                "delta_brier_vs_full_memory": _delta(with_memory_brier, no_reasoning_brier),
                "delta_ece_vs_baseline": _delta(baseline_ece, no_reasoning_ece),
                "delta_ece_vs_full_memory": _delta(with_memory_ece, no_reasoning_ece),
            },
        },
        "max_turns": effective_max_search_calls,
        "use_close_date_filter": effective_use_close_date_filter,
        "generated_at": _now_iso(),
    }
    summary_output_path = base_output_dir / "ablation_summary.json"
    _save_json(summary_output_path, ablation_summary)
    ablation_summary["summary_output_path"] = str(summary_output_path)
    return ablation_summary
