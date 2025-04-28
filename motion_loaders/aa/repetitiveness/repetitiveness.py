"""
Repetitiveness metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.repetitiveness.repetitiveness_planet import RepetitivenessPlanet

__main_class__ = "Repetitiveness"


class Repetitiveness(MetricAlias):
    """
    Repetitiveness metric superclass..
    """
    _SUBCLASS = RepetitivenessPlanet
