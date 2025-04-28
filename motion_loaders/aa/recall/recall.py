"""
Recall metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.recall.recall_planet import RecallPlanet

__main_class__ = "Recall"


class Recall(MetricAlias):
    """
    Recall metric superclass.
    """
    _SUBCLASS = RecallPlanet