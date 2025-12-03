"""Entraînement des modèles classiques"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import logging

from src.utils import save_model, logger

logger = logging.getLogger(__name__)


class FakeNewsDetector:
    """Détecteur de fake news"""
    
    def __init__(self, model_type='nb', calibrate: bool = True):
        self.model_type = model_type
        self.calibrate = calibrate
        self.model = self._create_model()
        self.is_trained = False
        self.is_calibrated = False
    
    def _create_model(self):
        models = {
            'nb': MultinomialNB(alpha=0.1),
            'svm': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        if self.model_type not in models:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
        
        logger.info(f"Modèle {self.model_type} créé")
        return models[self.model_type]
    
    def train(self, X_train, y_train):
        """Entraîner le modèle"""
        logger.info(f"Entraînement du modèle {self.model_type}...")
        
        from scipy.sparse import issparse
        if issparse(X_train) and self.model_type in ['rf', 'gb']:
            X_train = X_train.toarray()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True

        if self.calibrate and self.model_type in ['svm', 'lr']:
            try:
                logger.info(f"Calibration des probabilités...")
                calibrated = CalibratedClassifierCV(self.model, cv=3, method='sigmoid')
                calibrated.fit(X_train, y_train)
                self.model = calibrated
                self.is_calibrated = True
                logger.info("Calibration terminée")
            except Exception as e:
                logger.warning(f"Calibration échouée: {e}")
        
        logger.info("Entraînement terminé")
    
    def predict(self, X):
        """Prédire les labels"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        from scipy.sparse import issparse
        if issparse(X) and self.model_type in ['rf', 'gb']:
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Prédire les probabilités"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        from scipy.sparse import issparse
        if issparse(X) and self.model_type in ['rf', 'gb']:
            X = X.toarray()
        
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5) -> Dict[str, float]:
        """Validation croisée"""
        logger.info(f"Validation croisée avec {cv} folds...")
        
        from scipy.sparse import issparse
        if issparse(X) and self.model_type in ['rf', 'gb']:
            X = X.toarray()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"Précision moyenne: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        
        return results
    
    def save(self, filepath: str):
        """Sauvegarder le modèle"""
        save_model(self.model, filepath)
    
    def load(self, filepath: str):
        """
        Load model
        """
        from src.utils import load_model
        self.model = load_model(filepath)
        self.is_trained = True


class EnsembleDetector:
    """
    Ensemble detector using multiple models
    """
    
    def __init__(self, model_types=['nb', 'svm', 'rf']):
        """
        Initialize ensemble models
        """
        self.models = [FakeNewsDetector(model_type=mt) for mt in model_types]
        self.model_types = model_types
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train all models
        """
        
        for model in self.models:
            model.train(X_train, y_train)
        
        self.is_trained = True
    
    def predict(self, X, method='voting'):
        """Prédire par vote ou moyenne"""
        if not self.is_trained:
            raise ValueError("Modèles non entraînés")
        
        predictions = np.array([model.predict(X) for model in self.models])
        
        if method == 'voting':
            from scipy.stats import mode
            final_predictions = mode(predictions, axis=0)[0].flatten()
        elif method == 'averaging':
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_proba = probas.mean(axis=0)
            final_predictions = np.argmax(avg_proba, axis=1)
        else:
            raise ValueError(f"Méthode non supportée: {method}")
        
        return final_predictions
    
    def predict_proba(self, X):
        """Prédire probabilités (moyenne)"""
        if not self.is_trained:
            raise ValueError("Modèles non entraînés")
        
        probas = np.array([model.predict_proba(X) for model in self.models])
        return probas.mean(axis=0)


def train_multiple_models(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Entraîner et comparer plusieurs modèles"""
    from src.evaluation import evaluate_model
    
    model_types = ['nb', 'svm', 'rf', 'lr', 'gb']
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Entraînement modèle {model_type}")
        logger.info(f"{'='*50}")
        
        try:
            detector = FakeNewsDetector(model_type=model_type)
            detector.train(X_train, y_train)
            
            y_pred = detector.predict(X_test)
            y_pred_proba = detector.predict_proba(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            results[model_type] = {
                'model': detector,
                'metrics': metrics
            }
            
            logger.info(f"Précision {model_type}: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            continue
    
    return results


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X_train, y_train = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)
    X_test, y_test = make_classification(n_samples=200, n_features=100, n_classes=2, random_state=43)
    
    detector = FakeNewsDetector(model_type='nb')
    detector.train(X_train, y_train)
    
    y_pred = detector.predict(X_test)
