"""ConsensusRamen-v0 evidence-contract checks."""

import tempfile
import unittest
from pathlib import Path

from src.evaluation.evidence import (
    CONSENSUS_TRACE_FIELDS,
    JsonlTraceWriter,
)


def _trace_record():
    return {
        "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
        "ground_truth_class": 0, "prediction": 0, "correct": True,
        "predicted_entropy": 0.0, "inferred_context": None,
        "memory_size": 0, "num_active_contexts": None, "memory_bytes": 64,
        "latency_ms": 1.0,
    }


class ConsensusEvidenceTests(unittest.TestCase):
    def test_trace_writer_requires_a_complete_valid_consensus_group(self):
        consensus = {
            "consensus_mean_agreement": .8,
            "consensus_p10_agreement": .6,
            "consensus_p50_agreement": .8,
            "consensus_mask_rate": .5,
            "consensus_active_class_count": 3,
            "consensus_applied": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "consensus") as writer:
                with self.assertRaisesRegex(ValueError, "consensus trace fields"):
                    writer.write({**_trace_record(), "consensus_mean_agreement": .8})
                with self.assertRaisesRegex(ValueError, "consensus_mask_rate"):
                    writer.write({**_trace_record(), **consensus, "consensus_mask_rate": 1.1})
                with self.assertRaisesRegex(ValueError, "consensus_applied"):
                    writer.write({**_trace_record(), **consensus, "consensus_applied": 1})
                row = writer.write({**_trace_record(), **consensus})
        self.assertTrue(all(field in row for field in CONSENSUS_TRACE_FIELDS))

if __name__ == "__main__":
    unittest.main()
