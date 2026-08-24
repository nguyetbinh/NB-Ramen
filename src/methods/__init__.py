from .NoAdapt import NoAdapt
from .Tent import Tent
from .Ramen import Ramen
from .LatentRamen import LatentRamen
from .EntropyGatedLatentRamen import EntropyGatedLatentRamen
from .OracleLatentRamen import OracleLatentRamen
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
