from .NoAdapt import NoAdapt
from .Tent import Tent
from .Ramen import Ramen
from .LatentRamen import LatentRamen
from .EntropyGatedLatentRamen import EntropyGatedLatentRamen
from .OracleLatentRamen import OracleLatentRamen
from .OracleIDGradientRamen import OracleIDGradientRamen
from .OracleDropOODRamen import OracleDropOODRamen
from .OracleConsensusRamen import OracleConsensusRamen
from .ConsensusRamen import ConsensusRamen
# A separate method name selects the preregistered v1 YAML while deliberately
# reusing the same implementation.  It is an ablation identity, not a second
# deployable algorithm.
ConsensusRamenSoft = ConsensusRamen
ConsensusRamenNoSelf = ConsensusRamen
ConsensusRamenTau060 = ConsensusRamen
ConsensusRamenMin2 = ConsensusRamen
ConsensusRamenMin4 = ConsensusRamen
from .SupportAblations import (
    CausalRamen,
    ContextOnlyRamen,
    GlobalNearestRamen,
    RandomMemoryRamen,
    SameClassRamen,
)


def get_method_class(method_name):
    if method_name not in globals():
        raise NotImplementedError("Method not found: {}".format(method_name))
    return globals()[method_name]
