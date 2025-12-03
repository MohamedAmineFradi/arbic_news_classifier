"""Gestionnaire de modèles pour la détection de fake news"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import sys

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from src.data import ArabicTextPreprocessor
    from src.utils import load_model
    from config import MODELS_DIR
except ImportError as e:
    print(f"Erreur: Impossible d'importer les modules. ({e})")
    MODELS_DIR = Path("models")
    class ArabicTextPreprocessor:
        def preprocess(self, text): return text
    def load_model(path): return None

logger = logging.getLogger(__name__)

class ModelHandler:
    """Gestion des modèles ML pour la détection"""
    
    MODEL_NAMES = {
        'nb': 'Naïve Bayes',
        'svm': 'SVM (Support Vector Machine)',
        'lr': 'Régression Logistique',
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'arabert': 'AraBERT (Transformer)'
    }
    
    # Heuristiques inspirées de Detective Conan (indices de mensonge)
    SUSPICION_KEYWORDS = {
        # Mots exagérés
        'ديناصور', 'فضائي', 'فضائية', 'مخلوق غريب', 'كائن فضائي',
        # Prétentions magiques
        'معجزة', 'سحري', 'سحرية', 'خارق', 'خارقة للطبيعة',
        # Urgence excessive
        'عاجل جدا', 'عاجل وخطير', 'انتبه قبل فوات الأوان',
        # Catastrophisme
        'يوم القيامة', 'نهاية العالم', 'كارثة عالمية', 'دمار شامل',
        # Complots
        'تسريب خطير', 'يخفون عنك', 'الحكومة تخفي', 'مؤامرة كبرى',
        # Science-fiction
        'انفجار شمسي', 'حزام الكويكبات', 'بوابة زمنية',
        # Pseudo-science
        'التنبؤات', 'الأبراج تكشف', 'طاقة كونية', 'شاكرات',
        # Trop beau pour être vrai
        'يحول الماء إلى وقود', 'علاج سحري', 'يشفي كل الأمراض',
        'اربح المليون', 'مجاناً تماماً', 'بدون مجهود'
    }
    
    def __init__(self):
        self.preprocessor = ArabicTextPreprocessor()
        self.feature_extractor = None
        self.loaded_models: Dict[str, Any] = {}
        self.current_model_type = 'nb'
        
        self._load_feature_extractor()
    
    def _load_feature_extractor(self):
        """Charger l'extracteur de features TF-IDF"""
        try:
            extractor_path = MODELS_DIR / 'feature_extractor.pkl'
            if extractor_path.exists():
                self.feature_extractor = load_model(str(extractor_path))
                logger.info("✅ Extracteur de features chargé")
            else:
                logger.warning(f"⚠️ Extracteur introuvable: {extractor_path}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement extracteur: {e}")
    
    def load_model(self, model_key: str) -> bool:
        """
        Charger un modèle en mémoire cache
        
        Args:
            model_key: Clé du modèle (nb, svm, lr, rf, gb, arabert)
            
        Returns:
            True si chargement réussi, False sinon
        """
        # Déjà en cache
        if model_key in self.loaded_models:
            self.current_model_type = model_key
            logger.info(f"♻️ Modèle {model_key} déjà en cache")
            return True
        
        # Cas spécial: AraBERT
        if model_key == 'arabert':
            return self._load_arabert()
        
        # Modèles traditionnels ML
        try:
            model_path = MODELS_DIR / f"{model_key}_model.pkl"
            if not model_path.exists():
                logger.error(f"❌ Modèle introuvable: {model_path}")
                return False
            
            model = load_model(str(model_path))
            self.loaded_models[model_key] = model
            self.current_model_type = model_key
            logger.info(f"✅ Modèle {self.MODEL_NAMES[model_key]} chargé")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {model_key}: {e}")
            return False
    
    def _load_arabert(self) -> bool:
        """Charger le modèle AraBERT"""
        try:
            logger.info("🤖 Chargement AraBERT (peut prendre du temps)...")
            from src.models import AraBERTFakeNewsClassifier
            
            model = AraBERTFakeNewsClassifier()
            self.loaded_models['arabert'] = model
            self.current_model_type = 'arabert'
            logger.info("✅ AraBERT chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec chargement AraBERT: {e}")
            return False
    
    def _apply_conan_heuristics(
        self, 
        text: str, 
        prediction: int, 
        probabilities: List[float]
    ) -> Tuple[int, List[float]]:
        """
        Appliquer les heuristiques de Detective Conan
        
        Comme Conan qui détecte les indices subtils, cette fonction
        ajuste les prédictions basées sur des mots suspects
        
        Args:
            text: Texte original
            prediction: Prédiction initiale (0=fiable, 1=fake)
            probabilities: Probabilités [prob_fiable, prob_fake]
            
        Returns:
            Tuple (nouvelle_prediction, nouvelles_probabilites)
        """
        text_lower = text.lower()
        proba_diff = abs(probabilities[0] - probabilities[1])
        
        # Chercher des indices suspects
        suspicion_score = sum(
            1 for keyword in self.SUSPICION_KEYWORDS 
            if keyword in text_lower
        )
        
        # Si indices suspects ET faible confiance du modèle
        if suspicion_score > 0 and proba_diff < 0.25:
            logger.info(f"🔍 Conan détecte {suspicion_score} indice(s) suspect(s)")
            
            new_proba = probabilities.copy()
            
            # Augmenter la probabilité de fake news
            if new_proba[1] < 0.65:
                boost = min(0.15 * suspicion_score, 0.30)
                new_proba[1] = min(new_proba[1] + boost, 0.75)
                new_proba[0] = 1.0 - new_proba[1]
                
                return 1, new_proba
        
        return prediction, probabilities
    
    def predict(
        self, 
        text: str, 
        model_key: str
    ) -> Tuple[int, List[float], Dict[str, Any]]:
        """
        Effectuer une prédiction sur un texte
        
        Args:
            text: Texte à analyser
            model_key: Clé du modèle à utiliser
            
        Returns:
            Tuple (prediction, probabilites, metadata)
        """
        if not text or not text.strip():
            raise ValueError("Le texte est vide")
        
        # Charger le modèle si nécessaire
        if model_key not in self.loaded_models:
            success = self.load_model(model_key)
            if not success:
                raise RuntimeError(f"Impossible de charger le modèle {model_key}")
        
        model = self.loaded_models[model_key]
        
        # Prétraitement
        processed_text = self.preprocessor.preprocess(text)
        
        # Prédiction selon le type de modèle
        if model_key == 'arabert':
            predictions, probas = model.predict([processed_text])
            prediction = int(predictions[0])
            proba = probas[0]
        else:
            if self.feature_extractor is None:
                raise RuntimeError("Extracteur de features non disponible")
            
            X = self.feature_extractor.transform([processed_text])
            prediction = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0].tolist()
        
        # Appliquer les heuristiques de Conan
        final_pred, final_proba = self._apply_conan_heuristics(
            text, prediction, proba
        )
        
        # Métadonnées
        metadata = {
            'model': model_key,
            'model_name': self.MODEL_NAMES[model_key],
            'original_prediction': prediction,
            'adjusted_by_heuristics': (final_pred != prediction),
            'text_length': len(text),
            'word_count': len(text.split()),
            'processed_text_length': len(processed_text)
        }
        
        return final_pred, final_proba, metadata
    
    def get_available_models(self) -> List[str]:
        """Retourner la liste des modèles disponibles"""
        available = []
        
        # Vérifier les modèles traditionnels
        for key in ['nb', 'svm', 'lr', 'rf', 'gb']:
            model_path = MODELS_DIR / f"{key}_model.pkl"
            if model_path.exists():
                available.append(self.MODEL_NAMES[key])
        
        # AraBERT (toujours disponible si dépendances OK)
        try:
            import transformers
            available.append(self.MODEL_NAMES['arabert'])
        except ImportError:
            pass
        
        return available
