"""
Sacrebleu metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.sacrebleu.sacrebleu_planet import SacrebleuPlanet

__main_class__ = "Sacrebleu"


class Sacrebleu(MetricAlias):
    """
    Sacrebleu metric superclass.
    """
    _SUBCLASS = SacrebleuPlanet
