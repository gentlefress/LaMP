"""
NID metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.nid.nid_planet import NIDPlanet

__main_class__ = "NID"


class NID(MetricAlias):
    """
    NID metric super class.
    """
    _SUBCLASS = NIDPlanet
