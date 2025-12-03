"""Machine learning models module"""

from .classical_models import FakeNewsDetector, EnsembleDetector, train_multiple_models
from .arabert_detector import AraBERTFakeNewsClassifier

__all__ = [
    'FakeNewsDetector',
    'EnsembleDetector',
    'train_multiple_models',
    'AraBERTFakeNewsClassifier',
]
