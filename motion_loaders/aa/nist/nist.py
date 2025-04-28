"""
Nist metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.nist.nist_planet import NistPlanet

__main_class__ = "Nist"


class Nist(MetricAlias):
    """
    Nist metric superclass.
    """
    _SUBCLASS = NistPlanet
