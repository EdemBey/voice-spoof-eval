"""Rubric loading, aggregation and grading for the voice-spoof evaluation protocol.

The protocol is deliberately ordinal. A binary spoofed / not-spoofed label collapses
silence, a misparse, and a confidently wrong action into one bucket; the last of those
is the dangerous one, so it gets its own score.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: Ordinal scale used for every test case.
SCALE = {
    0: "No response at all.",
    1: "Response could not be understood or parsed.",
    2: "Understood but produced an incorrect or incomplete result.",
    3: "Fully correct and expected response.",
}

#: Grade thresholds applied to the cumulative average score.
GRADE_THRESHOLDS = ((2.7, "A"), (2.0, "B"), (1.0, "C"))


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    command: str
    description: str
    success_criteria: str


@dataclass(frozen=True)
class Observation:
    case_id: str
    platform: str
    attempts: int
    score: int

    def __post_init__(self) -> None:
        if self.score not in SCALE:
            raise ValueError(f"score must be one of {sorted(SCALE)}, got {self.score!r}")
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")


def grade(average: float) -> str:
    """Map a cumulative average score onto the project's letter grade."""
    for threshold, letter in GRADE_THRESHOLDS:
        if average >= threshold:
            return letter
    return "F"


def load_rubric(path: Path | None = None) -> List[Case]:
    path = path or DATA_DIR / "rubric.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return [Case(**row) for row in csv.DictReader(fh)]


def load_results(path: Path | None = None) -> List[Observation]:
    path = path or DATA_DIR / "results.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            Observation(
                case_id=row["id"],
                platform=row["platform"],
                attempts=int(row["attempts"]),
                score=int(row["score"]),
            )
            for row in csv.DictReader(fh)
        ]


def load_category_summary(path: Path | None = None) -> List[Dict[str, object]]:
    """Load the full 56-case score distribution, aggregated per category."""
    path = path or DATA_DIR / "category_summary.csv"
    rows: List[Dict[str, object]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: Dict[str, object] = {"category": row["category"]}
            for key, value in row.items():
                if key == "category":
                    continue
                parsed[key] = int(value)
            rows.append(parsed)
    return rows


def average_by_platform(observations: Iterable[Observation]) -> Dict[str, float]:
    totals: Dict[str, List[int]] = defaultdict(list)
    for obs in observations:
        totals[obs.platform].append(obs.score)
    return {platform: sum(scores) / len(scores) for platform, scores in totals.items()}


def average_by_category(
    observations: Sequence[Observation], rubric: Sequence[Case]
) -> Dict[str, Dict[str, float]]:
    category_of = {case.id: case.category for case in rubric}
    buckets: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for obs in observations:
        buckets[obs.platform][category_of[obs.case_id]].append(obs.score)
    return {
        platform: {cat: sum(v) / len(v) for cat, v in per_cat.items()}
        for platform, per_cat in buckets.items()
    }


def score_distribution(observations: Iterable[Observation]) -> Dict[str, Dict[int, int]]:
    dist: Dict[str, Dict[int, int]] = defaultdict(lambda: {s: 0 for s in SCALE})
    for obs in observations:
        dist[obs.platform][obs.score] += 1
    return {platform: dict(sorted(counts.items())) for platform, counts in dist.items()}


def summary_from_categories(category_rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    """Weighted aggregate over the full 56-case set, using the per-category distributions."""
    out: Dict[str, Dict[str, float]] = {}
    for prefix, label in (("siri", "Siri"), ("ga", "Google Assistant")):
        total_score = 0
        total_cases = 0
        per_category: Dict[str, float] = {}
        for row in category_rows:
            counts = [int(row[f"{prefix}_s{s}"]) for s in SCALE]
            n = sum(counts)
            if n == 0:
                continue
            weighted = sum(score * count for score, count in zip(SCALE, counts))
            per_category[str(row["category"])] = weighted / n
            total_score += weighted
            total_cases += n
        out[label] = {
            "per_category": per_category,
            "average": total_score / total_cases if total_cases else 0.0,
            "cases": float(total_cases),
        }
    return out


def format_markdown_table(
    per_category: Mapping[str, Mapping[str, float]],
    averages: Mapping[str, float],
) -> str:
    platforms = list(averages)
    categories = list(next(iter(per_category.values())).keys())
    header = "| Category | " + " | ".join(platforms) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in platforms) + " |"
    lines = [header, rule]
    for category in categories:
        cells = [
            f"{per_category[p].get(category, float('nan')):.2f}" for p in platforms
        ]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    totals = " | ".join(f"**{averages[p]:.2f}**" for p in platforms)
    grades = " | ".join(f"**{grade(averages[p])}**" for p in platforms)
    lines.append("| **Overall** | " + totals + " |")
    lines.append("| **Grade** | " + grades + " |")
    return "\n".join(lines)
