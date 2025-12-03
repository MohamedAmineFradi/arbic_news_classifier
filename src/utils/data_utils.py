"""Data manipulation utilities"""

import numpy as np
from typing import Dict, List
from .logging_config import logger


def calculate_text_statistics(texts: List[str]) -> Dict:
    """Calculer statistiques textuelles"""
    lengths = [len(text.split()) for text in texts]
    
    stats = {
        'عدد النصوص': len(texts),
        'متوسط عدد الكلمات': np.mean(lengths),
        'الانحراف المعياري': np.std(lengths),
        'الحد الأدنى': np.min(lengths),
        'الحد الأقصى': np.max(lengths),
        'الربع الأول': np.percentile(lengths, 25),
        'الوسيط': np.median(lengths),
        'الربع الثالث': np.percentile(lengths, 75),
    }
    
    return stats


def print_statistics(stats: Dict) -> None:
    """Afficher statistiques"""
    print("\n" + "="*50)
    print(" | Data Statistics")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("="*50 + "\n")


def split_data(X, y, test_size=0.2, random_state=42):
    """Diviser train/test"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    def _rows(obj):
        if hasattr(obj, 'shape'):
            return obj.shape[0]
        return len(obj)
    
    logger.info(f"Taille train: {_rows(X_train)}")
    logger.info(f"Taille test: {_rows(X_test)}")
    
    return X_train, X_test, y_train, y_test


def get_label_distribution(y) -> Dict:
    """Distribution des labels"""
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("\nDistribution des labels:")
    for label, count in distribution.items():
        percentage = (count / len(y)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    return distribution
