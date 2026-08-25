"""Runner selection contract for the preregistered soft consensus ablation."""

import unittest

from src.methods import ConsensusRamen, get_method_class


class ConsensusRamenSoftAliasTests(unittest.TestCase):
    def test_soft_ablation_name_reuses_consensus_implementation(self):
        self.assertIs(ConsensusRamen, get_method_class("ConsensusRamenSoft"))

    def test_no_self_ablation_name_reuses_consensus_implementation(self):
        self.assertIs(ConsensusRamen, get_method_class("ConsensusRamenNoSelf"))

    def test_tau_and_minimum_class_ablation_names_reuse_consensus_implementation(self):
        for name in ("ConsensusRamenTau060", "ConsensusRamenMin2", "ConsensusRamenMin4"):
            self.assertIs(ConsensusRamen, get_method_class(name))


if __name__ == "__main__":
    unittest.main()
