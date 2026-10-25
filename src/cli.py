"""Command line entry point.

    python -m src.cli score
    python -m src.cli demo
    python -m src.cli analyse reference.wav converted.wav
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src import failure_modes, rubric
from src.metrics import analyse, read_wav


def _format_value(value: float) -> str:
    if value != value:
        return "n/a"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    return f"{value:10.4f}"


def _synthetic_pair(
    sr: int = 48000, seconds: float = 2.0, mode: str = "octave"
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reference signal and a degraded 'converted' counterpart.

    The reference is a voiced 150 Hz source with a broadband noise floor. The converted
    version reproduces one of the artefacts the project actually observed:

    - ``octave``: pitch shifted up an octave, fewer harmonics, noise floor removed.
    - ``smear``:  phase-smeared harmonics, still fully intelligible to a listener but
      no longer periodic, so pitch tracking collapses.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 150.0 + 6.0 * np.sin(2 * np.pi * 0.8 * t)

    phase = 2 * np.pi * np.cumsum(f0) / sr
    reference = sum(np.sin(k * phase) / k for k in range(1, 8))
    reference += 0.15 * rng.standard_normal(reference.size)  # room noise floor
    reference /= np.max(np.abs(reference))

    converted = np.zeros_like(reference)
    if mode == "octave":
        for k in range(1, 5):
            converted += np.sin(k * 2.0 * phase) / k
    else:
        # Independent random-walk phase per harmonic: the harmonics stop being integer
        # multiples of a common fundamental, which is what destroys periodicity while
        # leaving the audio intelligible.
        for k in range(1, 8):
            walk = np.cumsum(rng.normal(0.0, 0.6, t.size))
            converted += np.sin(k * phase + walk) / k
    converted /= np.max(np.abs(converted))
    return reference.astype(np.float32), converted.astype(np.float32)


def cmd_score(_: argparse.Namespace) -> int:
    cases = rubric.load_rubric()
    observations = rubric.load_results()
    categories = rubric.load_category_summary()

    documented_avg = rubric.average_by_platform(observations)
    documented_by_cat = rubric.average_by_category(observations, cases)
    full = rubric.summary_from_categories(categories)

    print(f"Rubric cases defined: {len(cases)}")
    print(f"Documented observations: {len(observations)}\n")

    print("Documented cases only (n=33 per platform)")
    print(rubric.format_markdown_table(documented_by_cat, documented_avg))
    print()

    per_category = {p: data["per_category"] for p, data in full.items()}
    averages = {p: data["average"] for p, data in full.items()}
    print("Full protocol, per-category distribution (n=56 per platform)")
    print(rubric.format_markdown_table(per_category, averages))
    print()
    for platform, data in full.items():
        print(f"  {platform}: {data['average']:.2f} / 3.00 -> grade {rubric.grade(data['average'])}")

    print("\nScore distribution (documented cases)")
    for platform, dist in rubric.score_distribution(observations).items():
        total = sum(dist.values())
        parts = ", ".join(f"{score}:{count}" for score, count in dist.items())
        print(f"  {platform} (n={total}): {parts}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    sr = 48000
    reference, converted = _synthetic_pair(sr, mode=args.mode)
    features = analyse(reference, converted, sr)

    print(f"Synthetic reference vs '{args.mode}' conversion (no audio files needed)\n")
    print(f"  {'feature':32s} {'value':>10s}")
    print(f"  {'-' * 32} {'-' * 10}")
    for key, value in features.items():
        print(f"  {key:32s} {_format_value(value)}")

    flagged = failure_modes.assess(features)
    print("\nFailure modes consistent with these features:")
    print(failure_modes.describe(flagged))
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    reference, sr = read_wav(args.reference)
    converted, converted_sr = read_wav(args.converted)
    if converted_sr != sr:
        print(f"note: resampling converted audio from {converted_sr} Hz to {sr} Hz")

    features = analyse(reference, converted, sr)
    for key, value in features.items():
        print(f"  {key:32s} {_format_value(value)}")

    print("\nFailure modes consistent with these features:")
    print(failure_modes.describe(failure_modes.assess(features)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-spoof-eval",
        description="Evaluation harness for detecting AI-generated speech.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Aggregate the documented results and print grades.")
    score.set_defaults(func=cmd_score)

    demo = sub.add_parser("demo", help="Run the metrics on a synthetic reference/converted pair.")
    demo.add_argument(
        "--mode",
        choices=("octave", "smear"),
        default="octave",
        help="Artefact to simulate: pitch-shifted octave leap, or phase-smeared harmonics.",
    )
    demo.set_defaults(func=cmd_demo)

    analysis = sub.add_parser("analyse", help="Compare two WAV files.")
    analysis.add_argument("reference")
    analysis.add_argument("converted")
    analysis.set_defaults(func=cmd_analyse)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
