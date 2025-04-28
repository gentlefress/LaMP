"""
Mauve metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.mauve.mauve_planet import MauvePlanet

__main_class__ = "Mauve"


class Mauve(MetricAlias):
    """
    Mauve metric superclass.
    """
    _SUBCLASS = MauvePlanet
