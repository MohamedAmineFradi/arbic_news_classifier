"""Extraction de features textuelles"""

import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracteur de features"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.svd = None
        
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.8
            )
        elif self.method == 'binary':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.8,
                binary=True
            )
        else:
            raise ValueError(f"Méthode non supportée: {self.method}")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Entraîner et transformer"""
        logger.info(f"Extraction features avec {self.method}...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Forme matrice: {X.shape}")
        return X
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transformer nouveaux textes"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer non entraîné")
        return self.vectorizer.transform(texts)
    
    def apply_dimensionality_reduction(self, X, n_components=100):
        """Réduction dimensionnalité SVD"""
        logger.info(f"Réduction à {n_components} composants...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = self.svd.fit_transform(X)
        
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"Variance expliquée: {explained_variance:.4f}")
        
        return X_reduced
    
    def get_feature_names(self) -> List[str]:
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        else:
            return self.vectorizer.get_feature_names()
    
    def get_top_features(self, n_top=20) -> List[Tuple[str, float]]:
        """Top features"""
        if self.method != 'tfidf':
            logger.warning("Fonctionne mieux avec TF-IDF")
        
        feature_names = self.get_feature_names()
        
        if hasattr(self.vectorizer, 'idf_'):
            scores = self.vectorizer.idf_
            top_indices = np.argsort(scores)[-n_top:][::-1]
            top_features = [(feature_names[i], scores[i]) for i in top_indices]
            return top_features
        
        return []


class TextFeatureEngineering:
    """Engineering features supplémentaires"""
    
    @staticmethod
    def extract_statistical_features(texts: List[str]) -> np.ndarray:
        """Extraire features statistiques"""
        features = []
        
        for text in texts:
            words = text.split()
            chars = list(text)
            
            feature_dict = {
                'word_count': len(words),
                'char_count': len(chars),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'unique_word_ratio': len(set(words)) / len(words) if words else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_ratio': sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,
            }
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    @staticmethod
    def extract_sentiment_features(texts: List[str]) -> np.ndarray:
        """Extraire features sentiment"""
        positive_words = {'جيد', 'ممتاز', 'رائع', 'جميل', 'مفيد', 'نجح', 'فوز'}
        negative_words = {'سيء', 'فشل', 'كارثة', 'خطر', 'مشكلة', 'خسارة', 'ضرر'}
        
        features = []
        
        for text in texts:
            words = set(text.split())
            
            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))
            
            features.append([pos_count, neg_count, pos_count - neg_count])
        
        return np.array(features)
    
    @staticmethod
    def combine_features(*feature_matrices) -> np.ndarray:
        """Combiner matrices de features"""
        from scipy.sparse import hstack, issparse
        
        matrices = []
        for matrix in feature_matrices:
            if issparse(matrix):
                matrices.append(matrix)
            else:
                from scipy.sparse import csr_matrix
                matrices.append(csr_matrix(matrix))
        
        combined = hstack(matrices)
        return combined


if __name__ == "__main__":
    sample_texts = [
        "هذا خبر موثوق عن الأحداث الجارية",
        "شائعة كاذبة ومضللة للغاية",
        "تقرير إخباري دقيق ومحايد",
    ]
    
    extractor = FeatureExtractor(method='tfidf', max_features=100)
    X_tfidf = extractor.fit_transform(sample_texts)
    
    
    X_stats = TextFeatureEngineering.extract_statistical_features(sample_texts)
    
    X_combined = TextFeatureEngineering.combine_features(X_tfidf, X_stats)
