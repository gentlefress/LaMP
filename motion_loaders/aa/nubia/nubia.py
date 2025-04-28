"""
Nubia metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.nubia.nubia_planet import NubiaPlanet

__main_class__ = "Nubia"


class Nubia(MetricAlias):
    """
    Nubia metric superclass.
    """
    _SUBCLASS = NubiaPlanet
