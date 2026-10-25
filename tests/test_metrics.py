import unittest

import numpy as np

from src import failure_modes
from src.metrics import (
    analyse,
    cosine_similarity,
    dtw_distance,
    f0_autocorrelation,
    f0_distribution,
    high_freq_ratio,
    js_divergence,
    resample_linear,
    spectral_flatness,
    voiced_ratio,
)


def tone(freq: float = 150.0, sr: int = 48000, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class PitchTests(unittest.TestCase):
    def test_detects_known_pitch(self) -> None:
        sr = 48000
        f0 = f0_autocorrelation(tone(150.0, sr), sr)
        self.assertGreater(voiced_ratio(f0), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(f0)), 150.0, delta=3.0)

    def test_silence_is_unvoiced(self) -> None:
        sr = 48000
        f0 = f0_autocorrelation(np.zeros(sr, dtype=np.float32), sr)
        self.assertEqual(voiced_ratio(f0), 0.0)

    def test_noise_is_unvoiced(self) -> None:
        sr = 48000
        rng = np.random.default_rng(1)
        f0 = f0_autocorrelation(rng.standard_normal(sr).astype(np.float32), sr)
        self.assertLess(voiced_ratio(f0), 0.2)


class DistributionTests(unittest.TestCase):
    def test_js_divergence_bounds(self) -> None:
        p = f0_distribution(f0_autocorrelation(tone(150.0), 48000))
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=9)
        q = f0_distribution(f0_autocorrelation(tone(300.0), 48000))
        self.assertGreater(js_divergence(p, q), 0.1)
        self.assertLessEqual(js_divergence(p, q), 1.0)

    def test_empty_distribution_is_zero_vector(self) -> None:
        dist = f0_distribution(np.array([]))
        self.assertEqual(dist.sum(), 0.0)


class ContourTests(unittest.TestCase):
    def test_dtw_identity(self) -> None:
        series = np.linspace(100.0, 200.0, 50)
        self.assertAlmostEqual(dtw_distance(series, series), 0.0, places=9)

    def test_dtw_penalises_shift(self) -> None:
        # DTW warps time, so a constant offset is partly absorbed; it must still cost
        # more than matching a contour against itself.
        a = np.linspace(100.0, 200.0, 50)
        self.assertGreater(dtw_distance(a, a + 40.0), dtw_distance(a, a))

    def test_dtw_penalises_shape_change(self) -> None:
        a = np.linspace(100.0, 200.0, 50)
        reversed_tail = np.concatenate([np.linspace(100.0, 200.0, 25), np.linspace(200.0, 100.0, 25)])
        self.assertGreater(dtw_distance(a, reversed_tail), dtw_distance(a, a) + 10.0)

    def test_dtw_handles_unvoiced_gaps(self) -> None:
        a = np.array([100.0, np.nan, 120.0, np.nan, 140.0])
        self.assertTrue(np.isfinite(dtw_distance(a, a)))


class SimilarityTests(unittest.TestCase):
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=9)

    def test_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)

    def test_zero_vector(self) -> None:
        self.assertEqual(cosine_similarity(np.zeros(3), np.ones(3)), 0.0)


class SpectralTests(unittest.TestCase):
    def test_noise_has_high_flatness(self) -> None:
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(48000).astype(np.float32)
        self.assertGreater(spectral_flatness(noise, 48000), spectral_flatness(tone(), 48000))

    def test_high_freq_ratio(self) -> None:
        sr = 48000
        low = tone(150.0, sr)
        rng = np.random.default_rng(3)
        noisy = (low + 0.3 * rng.standard_normal(low.size)).astype(np.float32)
        self.assertGreater(high_freq_ratio(noisy, sr), high_freq_ratio(low, sr))

    def test_resample_preserves_duration(self) -> None:
        x = tone(150.0, 16000, 0.5)
        out = resample_linear(x, 16000, 48000)
        self.assertAlmostEqual(len(out) / 48000, 0.5, places=2)


class AnalysisTests(unittest.TestCase):
    def test_analyse_flags_harmonic_smearing(self) -> None:
        sr = 48000
        rng = np.random.default_rng(4)
        reference = tone(150.0, sr) + 0.01 * rng.standard_normal(sr)
        converted = rng.standard_normal(sr).astype(np.float32) * 0.1  # no pitch at all
        features = analyse(reference.astype(np.float32), converted, sr)
        keys = {mode.key for mode in failure_modes.assess(features)}
        self.assertIn("harmonic_smearing", keys)

    def test_analyse_reports_cosine_when_embeddings_given(self) -> None:
        sr = 48000
        a = tone(150.0, sr)
        features = analyse(a, a, sr, embedding_a=np.ones(8), embedding_b=np.ones(8))
        self.assertAlmostEqual(features["embedding_cosine"], 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
