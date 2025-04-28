"""
UNR metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.unr.unr_planet import UNRPlanet

__main_class__ = "UNR"


class UNR(MetricAlias):
    """
    UNR metric super class.
    """
    _SUBCLASS = UNRPlanet
