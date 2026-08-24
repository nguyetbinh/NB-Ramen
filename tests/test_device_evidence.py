import json
import unittest
from types import SimpleNamespace

from src.runtime.device_evidence import DeviceMemoryTracker, collect_hardware_evidence


class FakeCuda:
    def __init__(self):
        self.reset_devices = []
        self.peak = 8192

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_properties(self, index):
        self.properties_index = index
        return SimpleNamespace(name="Mock GPU", total_memory=16_384)

    def get_device_capability(self, index):
        return (8, 6)

    def reset_peak_memory_stats(self, device):
        self.reset_devices.append(device)

    def max_memory_allocated(self, device):
        return self.peak


class FakeMps:
    def __init__(self, samples):
        self.samples = iter(samples)

    def current_allocated_memory(self):
        return next(self.samples)

    def driver_allocated_memory(self):
        return 300

    def recommended_max_memory(self):
        return 400


class DeviceEvidenceTests(unittest.TestCase):
    @staticmethod
    def torch(cuda=None, mps=None):
        return SimpleNamespace(
            __version__="mock-torch", cuda=cuda,
            mps=mps,
            version=SimpleNamespace(cuda="12.1"),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: 9010),
                mps=SimpleNamespace(
                    is_built=lambda: mps is not None,
                    is_available=lambda: mps is not None,
                ),
            ),
            get_num_threads=lambda: 4,
            get_num_interop_threads=lambda: 2,
        )

    def test_cpu_evidence_and_memory_are_json_safe_and_not_applicable(self):
        evidence = collect_hardware_evidence("cpu", torch_module=self.torch())
        self.assertEqual("cpu", evidence["requested_device"])
        self.assertEqual(4, evidence["torch"]["num_threads"])
        self.assertFalse(evidence["cuda"]["available"])
        self.assertEqual(
            {"status": "not_applicable", "kind": "unsupported", "bytes": None},
            DeviceMemoryTracker("cpu", torch_module=self.torch()).summary(),
        )
        json.dumps(evidence)

    def test_cuda_evidence_and_tracker_report_exact_allocator_peak(self):
        cuda = FakeCuda()
        device = SimpleNamespace(type="cuda", index=1)
        evidence = collect_hardware_evidence(device, torch_module=self.torch(cuda=cuda))
        self.assertEqual(1, evidence["cuda"]["device_index"])
        self.assertEqual("Mock GPU", evidence["cuda"]["name"])
        self.assertEqual([8, 6], evidence["cuda"]["capability"])
        tracker = DeviceMemoryTracker(device, torch_module=self.torch(cuda=cuda))
        tracker.start()
        self.assertEqual([device], cuda.reset_devices)
        self.assertEqual(
            {"status": "collected", "kind": "exact_cuda_allocator_peak", "bytes": 8192},
            tracker.summary(),
        )

    def test_mps_evidence_and_tracker_are_explicitly_sampled(self):
        mps = FakeMps([100, 250, 175, 275])
        torch = self.torch(mps=mps)
        evidence = collect_hardware_evidence("mps", torch_module=torch)
        self.assertTrue(evidence["mps"]["built"])
        self.assertTrue(evidence["mps"]["available"])
        self.assertEqual(100, evidence["mps"]["current_allocated_memory_bytes"])
        tracker = DeviceMemoryTracker("mps", torch_module=torch)
        tracker.sample_post_batch()
        tracker.sample_post_batch()
        tracker.sample_post_batch()
        self.assertEqual(
            {"status": "sampled", "kind": "sampled_post_batch_maximum", "bytes": 275},
            tracker.summary(),
        )


if __name__ == "__main__":
    unittest.main()
