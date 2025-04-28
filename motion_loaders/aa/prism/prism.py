"""
Prism metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.prism.prism_planet import PrismPlanet

__main_class__ = "Prism"


class Prism(MetricAlias):
    """
    Prism metric superclass.
    """
    _SUBCLASS = PrismPlanet
