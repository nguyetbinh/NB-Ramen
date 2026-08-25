"""Oracle Ramen control that prevents evaluator-labelled OOD items entering memory."""

from .OracleIDGradientRamen import OracleIDGradientRamen


class OracleDropOODRamen(OracleIDGradientRamen):
    """ID-only gradient oracle with OOD samples dropped before cache insertion."""

    drop_ood_from_memory = True
