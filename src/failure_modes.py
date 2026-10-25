"""Machine-readable taxonomy of the failure modes observed in the protocol.

The point of a taxonomy is that the next person does not have to rediscover the same
failures. Each entry states what you see, which metric moves, and what it means for a
system that has to make a decision from the audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FailureMode:
    key: str
    label: str
    symptom: str
    signal: str
    consequence: str


FAILURE_MODES: Dict[str, FailureMode] = {
    "harmonic_smearing": FailureMode(
        key="harmonic_smearing",
        label="Harmonic smearing",
        symptom="Audio is fully intelligible but pitch tracking returns almost no voiced frames.",
        signal="voiced_ratio_converted collapses towards 0 while f0_mean_reference is normal.",
        consequence="A detector built on F0 continuity silently treats synthetic audio as silence.",
    ),
    "octave_leap": FailureMode(
        key="octave_leap",
        label="Octave leap",
        symptom="Converted pitch lands an octave above the plausible range for the target speaker.",
        signal="f0_mean_converted is roughly 2x f0_mean_reference (e.g. 411 Hz on a male target).",
        consequence="Trivially flaggable if the enrolled speaker has a bounded pitch range.",
    ),
    "codec_masking": FailureMode(
        key="codec_masking",
        label="Codec masking",
        symptom="Compression strips the high-frequency artefacts detectors rely on.",
        signal="high_freq_ratio_converted and spectral_flatness_converted both drop after transcoding.",
        consequence="Acoustic-only countermeasures degrade exactly as the channel gets realistic.",
    ),
    "embedding_robustness": FailureMode(
        key="embedding_robustness",
        label="Embedding robustness under compression",
        symptom="Speaker verification still accepts the clone after a lossy round trip.",
        signal="embedding_cosine stays high (>0.8) against the enrolled speaker post-transcode.",
        consequence="Voice biometrics alone are not a control. Identity match is necessary, not sufficient.",
    ),
    "wakeword_gate": FailureMode(
        key="wakeword_gate",
        label="Wake-word gate dominance",
        symptom="The assistant rejects every clone at activation but accepts commands once triggered.",
        signal="Near-zero Activation scores next to high Information Retrieval scores on the same platform.",
        consequence="Perceived security comes from the gate, not the speaker model. Workflows without a gate inherit nothing.",
    ),
    "duration_dependence": FailureMode(
        key="duration_dependence",
        label="Duration dependence",
        symptom="Short clips are inconclusive; verdicts stabilise only with several seconds of audio.",
        signal="Score variance falls as utterance length increases.",
        consequence="Real-time detection has a latency budget, and two seconds of audio may not be enough.",
    ),
}


def assess(features: Dict[str, float]) -> List[FailureMode]:
    """Flag the failure modes that the supplied metrics are consistent with."""
    hits: List[FailureMode] = []

    def value(key: str, default: float = float("nan")) -> float:
        v = features.get(key, default)
        return float(v) if v is not None else default

    voiced_ref = value("voiced_ratio_reference", 0.0)
    voiced_conv = value("voiced_ratio_converted", 0.0)
    f0_ref = value("f0_mean_reference")
    f0_conv = value("f0_mean_converted")

    if voiced_ref > 0.5 and voiced_conv < 0.2:
        hits.append(FAILURE_MODES["harmonic_smearing"])

    if f0_ref == f0_ref and f0_conv == f0_conv and f0_ref > 0:
        ratio = f0_conv / f0_ref
        if 1.8 <= ratio <= 2.2 or 0.45 <= ratio <= 0.55:
            hits.append(FAILURE_MODES["octave_leap"])

    hf_ref = value("high_freq_ratio_reference")
    hf_conv = value("high_freq_ratio_converted")
    if hf_ref == hf_ref and hf_conv == hf_conv and hf_ref > 0:
        if hf_conv / hf_ref < 0.6:
            hits.append(FAILURE_MODES["codec_masking"])

    cosine = features.get("embedding_cosine")
    if cosine is not None and float(cosine) > 0.8:
        hits.append(FAILURE_MODES["embedding_robustness"])

    return hits


def describe(modes: Sequence[FailureMode]) -> str:
    if not modes:
        return "No known failure mode matched the supplied features."
    lines = []
    for mode in modes:
        lines.append(f"- {mode.label}: {mode.symptom} [{mode.signal}]")
    return "\n".join(lines)
