import unittest
from pathlib import Path

from src import rubric


class GradeTests(unittest.TestCase):
    def test_thresholds(self) -> None:
        self.assertEqual(rubric.grade(2.70), "A")
        self.assertEqual(rubric.grade(2.69), "B")
        self.assertEqual(rubric.grade(2.00), "B")
        self.assertEqual(rubric.grade(1.99), "C")
        self.assertEqual(rubric.grade(1.00), "C")
        self.assertEqual(rubric.grade(0.99), "F")


class RubricTests(unittest.TestCase):
    def test_dataset_shapes(self) -> None:
        cases = rubric.load_rubric()
        observations = rubric.load_results()
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(observations), 66)

    def test_ids_are_unique(self) -> None:
        cases = rubric.load_rubric()
        self.assertEqual(len({case.id for case in cases}), len(cases))

    def test_every_observation_has_a_case(self) -> None:
        ids = {case.id for case in rubric.load_rubric()}
        for obs in rubric.load_results():
            self.assertIn(obs.case_id, ids)

    def test_documented_averages(self) -> None:
        cases = rubric.load_rubric()
        observations = rubric.load_results()
        averages = rubric.average_by_platform(observations)
        self.assertAlmostEqual(averages["Siri"], 2.00, places=2)
        self.assertAlmostEqual(averages["Google Assistant"], 1.36, places=2)

        per_category = rubric.average_by_category(observations, cases)
        self.assertAlmostEqual(per_category["Siri"]["Activation"], 2.25, places=2)
        self.assertAlmostEqual(per_category["Google Assistant"]["Activation"], 0.00, places=2)
        self.assertAlmostEqual(per_category["Siri"]["Security & Privacy"], 2.875, places=2)

    def test_full_protocol_aggregation(self) -> None:
        full = rubric.summary_from_categories(rubric.load_category_summary())
        self.assertEqual(full["Siri"]["cases"], 56)
        self.assertEqual(full["Google Assistant"]["cases"], 56)
        self.assertAlmostEqual(full["Siri"]["average"], 1.98, places=2)
        self.assertAlmostEqual(full["Google Assistant"]["per_category"]["Activation"], 0.00, places=2)
        self.assertAlmostEqual(full["Siri"]["per_category"]["Media"], 3.00, places=2)

    def test_distribution_totals(self) -> None:
        dist = rubric.score_distribution(rubric.load_results())
        for platform, counts in dist.items():
            self.assertEqual(sum(counts.values()), 33, platform)
            self.assertEqual(sorted(counts), [0, 1, 2, 3])


class ObservationValidationTests(unittest.TestCase):
    def test_rejects_invalid_score(self) -> None:
        with self.assertRaises(ValueError):
            rubric.Observation(case_id="A01", platform="Siri", attempts=1, score=4)

    def test_rejects_zero_attempts(self) -> None:
        with self.assertRaises(ValueError):
            rubric.Observation(case_id="A01", platform="Siri", attempts=0, score=3)


if __name__ == "__main__":
    unittest.main()
