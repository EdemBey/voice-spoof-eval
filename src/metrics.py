"""Acoustic metrics for comparing a reference recording against a converted one.

Everything here runs on numpy alone: no model weights, no deep learning, no audio
framework. The point is that cheap, explainable features already separate synthetic
speech from real speech in the conditions this project looked at. Embeddings are what
tell you whether a *specific* identity was cloned; these features tell you whether the
signal is physically plausible at all.
"""

from __future__ import annotations

import wave
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "read_wav",
    "resample_linear",
    "frames",
    "f0_autocorrelation",
    "voiced_ratio",
    "f0_distribution",
    "js_divergence",
    "dtw_distance",
    "cosine_similarity",
    "high_freq_ratio",
    "spectral_flatness",
    "analyse",
]


# ── I/O ──────────────────────────────────────────────────────────────────────


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV file as mono float32 in [-1, 1] plus its sample rate."""
    with wave.open(path, "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Lightweight linear resampler. Adequate for feature work, not for playback."""
    if src_rate == dst_rate:
        return x
    duration = len(x) / src_rate
    target_len = int(round(duration * dst_rate))
    src_idx = np.linspace(0.0, len(x) - 1, num=len(x))
    dst_idx = np.linspace(0.0, len(x) - 1, num=target_len)
    return np.interp(dst_idx, src_idx, x).astype(np.float32)


def frames(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split a signal into overlapping frames, shape (n_frames, frame_len)."""
    if len(x) < frame_len:
        return np.empty((0, frame_len), dtype=x.dtype)
    count = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


# ── Pitch ────────────────────────────────────────────────────────────────────


def f0_autocorrelation(
    x: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    voicing_threshold: float = 0.30,
) -> np.ndarray:
    """Per-frame fundamental frequency via normalised autocorrelation.

    Returns one value per frame; ``nan`` marks an unvoiced or silent frame. A frame is
    considered unvoiced when the strongest autocorrelation peak in the plausible pitch
    range falls below ``voicing_threshold`` — which is exactly what happens to vocoder
    output whose harmonics have been smeared away.
    """
    x = np.asarray(x, dtype=np.float64)
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    block = frames(x, frame_len, hop)
    if block.size == 0:
        return np.array([], dtype=float)

    block = block - block.mean(axis=1, keepdims=True)
    block *= np.hanning(frame_len)[None, :]

    min_lag = max(1, int(sr / fmax))
    max_lag = min(frame_len - 2, int(sr / fmin))
    out = np.full(block.shape[0], np.nan)

    for i, frame in enumerate(block):
        energy = float(np.dot(frame, frame))
        if energy <= 1e-12:
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if ac[0] <= 0:
            continue
        ac /= ac[0]

        segment = ac[min_lag : max_lag + 1]
        if segment.size == 0:
            continue
        peak_offset = int(np.argmax(segment))
        if segment[peak_offset] < voicing_threshold:
            continue

        lag = float(min_lag + peak_offset)
        # Parabolic interpolation around the peak for sub-sample precision.
        if 0 < lag < len(ac) - 1:
            left, centre, right = ac[int(lag) - 1], ac[int(lag)], ac[int(lag) + 1]
            denom = left - 2.0 * centre + right
            if abs(denom) > 1e-12:
                lag += 0.5 * (left - right) / denom
        if lag > 0:
            out[i] = sr / lag
    return out


def voiced_ratio(f0: np.ndarray) -> float:
    """Share of frames that carry a detectable pitch."""
    if f0.size == 0:
        return 0.0
    return float(np.mean(~np.isnan(f0)))


# ── Distributions ─────────────────────────────────────────────────────────────


def f0_distribution(
    f0: np.ndarray, fmin: float = 60.0, fmax: float = 400.0, bins: int = 64
) -> np.ndarray:
    """Normalised histogram of voiced F0 values, clipped to the plausible range."""
    voiced = f0[~np.isnan(f0)] if f0.size else np.array([])
    if voiced.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(np.clip(voiced, fmin, fmax), bins=bins, range=(fmin, fmax))
    total = hist.sum()
    return hist / total if total else hist.astype(float)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size != q.size:
        raise ValueError("distributions must have the same length")
    p = p / p.sum() if p.sum() else np.full_like(p, 1.0 / p.size)
    q = q / q.sum() if q.sum() else np.full_like(q, 1.0 / q.size)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ── Contour comparison ────────────────────────────────────────────────────────


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs so DTW can run on the voiced portion."""
    values = np.asarray(values, dtype=float)
    mask = np.isnan(values)
    if mask.all():
        return np.array([], dtype=float)
    if not mask.any():
        return values
    idx = np.arange(values.size)
    values = values.copy()
    values[mask] = np.interp(idx[mask], idx[~mask], values[~mask])
    return values


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute DTW cost between two pitch contours, normalised by length."""
    a = _fill_gaps(a)
    b = _fill_gaps(b)
    if a.size == 0 or b.size == 0:
        return float("inf")

    n, m = a.size, b.size
    cost = np.abs(a[:, None] - b[None, :])
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return float(acc[n, m] / max(n, m))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("vectors must have the same length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Spectral shape ────────────────────────────────────────────────────────────


def high_freq_ratio(x: np.ndarray, sr: int, cutoff: float = 6000.0) -> float:
    """Share of spectral energy above ``cutoff``.

    Real recordings carry a noise floor and breath energy up here. Vocoder output tends
    to be clean above the modelled band, so this ratio collapses even when the audio
    sounds perfectly natural on a laptop speaker.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(spectrum[freqs >= cutoff].sum() / total)


def spectral_flatness(x: np.ndarray, sr: int, frame_ms: float = 40.0) -> float:
    """Mean spectral flatness (geometric / arithmetic mean of power spectrum).

    Near 1.0 means noise-like (a real room), near 0.0 means tonal (a vocoder).
    """
    x = np.asarray(x, dtype=float)
    frame_len = int(sr * frame_ms / 1000.0)
    block = frames(x, frame_len, frame_len // 2)
    if block.size == 0:
        return 0.0
    block = block * np.hanning(frame_len)[None, :]
    power = np.abs(np.fft.rfft(block, axis=1)) ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(power), axis=1))
    arith = np.mean(power, axis=1)
    return float(np.mean(geo / arith))


# ── Bundle ────────────────────────────────────────────────────────────────────


def analyse(
    reference: np.ndarray,
    converted: np.ndarray,
    sr: int,
    embedding_a: Optional[np.ndarray] = None,
    embedding_b: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run the full feature set on a reference / converted pair."""
    converted = resample_linear(converted, sr, sr)
    f0_ref = f0_autocorrelation(reference, sr)
    f0_conv = f0_autocorrelation(converted, sr)

    result = {
        "voiced_ratio_reference": voiced_ratio(f0_ref),
        "voiced_ratio_converted": voiced_ratio(f0_conv),
        "f0_mean_reference": float(np.nanmean(f0_ref)) if voiced_ratio(f0_ref) else float("nan"),
        "f0_mean_converted": float(np.nanmean(f0_conv)) if voiced_ratio(f0_conv) else float("nan"),
        "f0_js_divergence": js_divergence(f0_distribution(f0_ref), f0_distribution(f0_conv)),
        "f0_dtw_distance": dtw_distance(f0_ref, f0_conv),
        "high_freq_ratio_reference": high_freq_ratio(reference, sr),
        "high_freq_ratio_converted": high_freq_ratio(converted, sr),
        "spectral_flatness_reference": spectral_flatness(reference, sr),
        "spectral_flatness_converted": spectral_flatness(converted, sr),
    }
    if embedding_a is not None and embedding_b is not None:
        result["embedding_cosine"] = cosine_similarity(embedding_a, embedding_b)
    return result
