"""
Perplexity metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.perplexity.perplexity_planet import PerplexityPlanet

__main_class__ = "Perplexity"


class Perplexity(MetricAlias):
    """
    Perplexity metric super class.
    """
    _SUBCLASS = PerplexityPlanet
