"""
Meteor metric super class.
"""
from nlgmetricverse.metrics._core import MetricAlias
from nlgmetricverse.metrics.meteor.meteor_planet import MeteorPlanet

__main_class__ = "Meteor"


class Meteor(MetricAlias):
    """
    Meteor metric superclass.
    """
    _SUBCLASS = MeteorPlanet
