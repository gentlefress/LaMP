"""
TER metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.ter.ter_planet import TERPlanet

__main_class__ = "TER"


class TER(MetricAlias):
    """
    TER metric super class.
    """
    _SUBCLASS = TERPlanet
