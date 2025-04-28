"""
WMD metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.wmd.wmd_planet import WMDPlanet

__main_class__ = "WMD"


class WMD(MetricAlias):
    """
    WMD metric superclass.
    """
    _SUBCLASS = WMDPlanet
