import unittest
from bytebanklm.planner import Job, plan_jobs


class PlannerTests(unittest.TestCase):
    def job(self, name: str, weight: int, priority: int = 0) -> Job:
        return Job(name, weight, context=10, layers=2, kv_heads=2, head_dim=4, kv_element_bytes=2, priority=priority)

    def test_kv_formula_counts_keys_and_values(self) -> None:
        self.assertEqual(self.job("a", 100).kv_cache_bytes, 640)

    def test_priority_controls_deterministic_admission(self) -> None:
        low = self.job("low", 700, 0)
        high = self.job("high", 700, 10)
        plan = plan_jobs(2000, 500, [low, high])
        self.assertEqual([item.name for item in plan.decisions], ["high", "low"])
        self.assertTrue(plan.decisions[0].accepted)
        self.assertFalse(plan.decisions[1].accepted)
        self.assertEqual(plan.remaining_bytes, 160)

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            plan_jobs(4000, 0, [self.job("same", 1), self.job("same", 1)])

if __name__ == "__main__":
    unittest.main()
