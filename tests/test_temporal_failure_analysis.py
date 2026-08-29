import unittest

from src.evaluation.temporal_failure_analysis import annotate_time_since_shift, paired_panel_series, temporal_failure_report


def row(timestep, domain, base, adapted, **extra):
    return {"timestep": timestep, "ground_truth_domain": domain, "base_correct": base, "adapted_correct": adapted,
            "seed": 1, "stream": "block", "memory_occupancy": timestep, **extra}


class TemporalFailureTests(unittest.TestCase):
    def test_shift_boundaries_and_paired_panels(self):
        rows = [row(0, "a", True, True, consensus_mean=.8), row(1, "a", False, True, consensus_mean=.7),
                row(2, "b", True, False, consensus_mean=.2), row(3, "b", False, False, consensus_mean=.1)]
        annotated = annotate_time_since_shift(rows)
        self.assertEqual([0, 1, 0, 1], [item["time_since_shift"] for item in annotated])
        report = temporal_failure_report(rows, timestep_bin_size=2, minimum_count=2)
        self.assertEqual("computed", report["status"])
        self.assertEqual(.5, report["paired_panels"]["series"][0]["task_failure"]["base_error"])
        self.assertEqual("computed", report["paired_panels"]["series"][0]["mechanism"]["consensus_mean"]["status"])

    def test_empty_and_small_strata(self):
        self.assertEqual("insufficient", temporal_failure_report([])["status"])
        result = temporal_failure_report([row(0, "a", True, True)], minimum_count=2)
        self.assertEqual("insufficient", result["strata"]["domain"][0]["status"])
        panels = paired_panel_series([row(0, "a", True, True)], minimum_count=2)
        self.assertEqual("insufficient", panels["series"][0]["status"])


if __name__ == "__main__": unittest.main()
