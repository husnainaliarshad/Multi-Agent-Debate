"""
Hugging Face `K-and-K/knights-and-knaves` helpers for benchmarking debates.

Each row contains a `quiz` (puzzle text) and `solution_text` (gold answer).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TOPIC_SUFFIX = (
    "\n\nProvide a step-by-step deduction if helpful, and end with an explicit "
    "classification for every inhabitant named (who is a knight and who is a knave)."
)


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `datasets` package is required for Knights & Knaves benchmarks. "
            "Install backend dependencies (e.g. pip install datasets)."
        ) from exc
    return load_dataset


def load_knk_split(config_name: str = "test", split: str = "2ppl"):
    load_dataset = _require_datasets()
    return load_dataset("K-and-K/knights-and-knaves", config_name, split=split)


def slice_dataset(
    ds,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
):
    if shuffle:
        ds = ds.shuffle(seed=42 if seed is None else int(seed))
    start = max(0, int(offset))
    n = len(ds)
    end = n if limit is None else min(n, start + int(limit))
    if start >= n:
        return ds.select([])
    return ds.select(range(start, end))


def format_debate_topic(quiz: str, *, add_suffix: bool = True) -> str:
    text = (quiz or "").strip()
    if add_suffix and DEFAULT_TOPIC_SUFFIX.strip() not in text:
        text = f"{text}{DEFAULT_TOPIC_SUFFIX}"
    return text


def rows_to_topics_and_gold(
    rows: List[Dict[str, Any]],
    *,
    add_topic_suffix: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    gold: List[Dict[str, Any]] = []
    topics: List[str] = []
    for row in rows:
        idx = row.get("index")
        quiz = row.get("quiz", "")
        solution_text = row.get("solution_text", "")
        names = row.get("names")
        solution = row.get("solution")
        gold.append(
            {
                "index": idx,
                "quiz": quiz,
                "solution_text": solution_text,
                "names": names,
                "solution": solution,
            }
        )
        topics.append(format_debate_topic(quiz, add_suffix=add_topic_suffix))
    return gold, topics


def build_experiment_payload(
    *,
    config_name: str = "test",
    split: str = "2ppl",
    offset: int = 0,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
    add_topic_suffix: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = load_knk_split(config_name=config_name, split=split)
    ds = slice_dataset(ds, offset=offset, limit=limit, shuffle=shuffle, seed=seed)
    rows = [ds[i] for i in range(len(ds))]
    return rows_to_topics_and_gold(rows, add_topic_suffix=add_topic_suffix)


def preview_rows(
    config_name: str = "test",
    split: str = "2ppl",
    limit: int = 5,
    offset: int = 0,
) -> Dict[str, Any]:
    ds = load_knk_split(config_name=config_name, split=split)
    window = slice_dataset(ds, offset=offset, limit=limit, shuffle=False, seed=None)
    items = []
    for i in range(len(window)):
        row = window[i]
        items.append(
            {
                "index": row.get("index"),
                "topic_preview": (row.get("quiz") or "")[:280],
                "solution_text": row.get("solution_text"),
            }
        )
    return {
        "dataset": "K-and-K/knights-and-knaves",
        "config_name": config_name,
        "split": split,
        "total_rows": len(ds),
        "preview_count": len(items),
        "items": items,
    }


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _expected_assignments(
    gold_solution_text: str,
    names: Optional[List[str]] = None,
    solution: Optional[List[bool]] = None,
) -> Dict[str, str]:
    if names and solution and len(names) == len(solution):
        return {
            str(name).lower(): "knight" if bool(value) else "knave"
            for name, value in zip(names, solution)
            if str(name).strip()
        }

    assignments: Dict[str, str] = {}
    for match in re.finditer(
        r"\b([A-Z][A-Za-z'-]*)\s+is\s+(?:a|an)\s+(knight|knave)\b",
        gold_solution_text or "",
        flags=re.IGNORECASE,
    ):
        assignments[match.group(1).lower()] = match.group(2).lower()
    return assignments


def _answer_region(text: str) -> str:
    markers = [
        "final verdict",
        "final answer",
        "conclusion",
        "classification",
        "answer:",
    ]
    lowered = text.lower()
    for marker in markers:
        start = lowered.rfind(marker)
        if start >= 0:
            return text[start:]
    return text


def _extract_role_for_name(text: str, name: str) -> Optional[str]:
    escaped_name = re.escape(name)
    patterns = [
        rf"\b{escaped_name}\b\s*(?:is|=|:)\s*(?:a|an|the)?\s*(knight|knave)\b",
        rf"\b{escaped_name}\b[^.\n;:]*?\b(?:is|as|be|being)\s+(?:a|an|the)?\s*(knight|knave)\b",
        rf"\b{escaped_name}\b\s+(?:the\s+)?(knight|knave)\b",
        rf"\b(knights|knaves)\b\s*[:\-]\s*[^.\n]*?\b{escaped_name}\b",
    ]

    matches: List[Tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            role = match.group(1).lower().rstrip("s")
            matches.append((match.start(), role))
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[-1][1]


def verdict_matches_gold(
    judge_text: str,
    gold_solution_text: str,
    names: Optional[List[str]] = None,
    solution: Optional[List[bool]] = None,
) -> bool:
    """
    Match K&K answers by role assignment, with the exact gold sentence as a fast path.

    The dataset exposes structured `names` + boolean `solution` fields. Use those
    when available so correct variants like "Zoey is a Knave (K)" are not marked
    wrong merely because they do not repeat `solution_text` verbatim.
    """
    if not judge_text or not gold_solution_text:
        return False
    gold_n = _normalize(gold_solution_text)
    judge_n = _normalize(judge_text)
    if len(gold_n) >= 8 and gold_n in judge_n:
        return True

    expected = _expected_assignments(gold_solution_text, names=names, solution=solution)
    if not expected:
        return False

    answer_text = _answer_region(judge_text)
    for name, expected_role in expected.items():
        observed_role = _extract_role_for_name(answer_text, name)
        if observed_role is None and answer_text != judge_text:
            observed_role = _extract_role_for_name(judge_text, name)
        if observed_role != expected_role:
            return False
    return True
