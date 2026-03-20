"""Aegis behavioral classifiers for alignment auditing.

Re-exports all classifier classes for convenient access:

    from classifiers import SpecGamingClassifier, ClassifierPipeline
"""

from .base import BaseClassifier, ClassifierResult, ClassifierPipeline
from .spec_gaming import SpecGamingClassifier
from .deceptive_alignment import DeceptiveAlignmentClassifier
from .capability_elicitation import CapabilityElicitationClassifier
from .corrigibility import CorrigibilityClassifier

__all__ = [
    # Base
    "BaseClassifier",
    "ClassifierResult",
    "ClassifierPipeline",
    # Classifiers
    "SpecGamingClassifier",
    "DeceptiveAlignmentClassifier",
    "CapabilityElicitationClassifier",
    "CorrigibilityClassifier",
]
