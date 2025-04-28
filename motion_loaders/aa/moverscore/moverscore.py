"""
Moverscore metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.moverscore.moverscore_planet import MoverscorePlanet

__main_class__ = "Moverscore"


class Moverscore(MetricAlias):
    """
    Moverscore metric superclass.
    """
    _SUBCLASS = MoverscorePlanet
