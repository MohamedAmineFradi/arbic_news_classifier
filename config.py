"""Configuration du projet"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
PREPROCESSING_CONFIG = {
    'remove_diacritics': True,
    'remove_punctuation': True,
    'remove_english': True,
    'remove_numbers': False,
    'normalize_arabic': True,
    'remove_stopwords': True,
    'apply_stemming': True,
}

MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'max_features': 5000,
}

ARABERT_CONFIG = {
    'model_name': 'aubmindlab/bert-base-arabertv2',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
}

CLASSES = {
    0: 'موثوقة',
    1: 'مضللة',
}

CUSTOM_STOPWORDS = [
    'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي',
    'كان', 'قال', 'إن', 'لم', 'قد', 'كما', 'عن', 'بعد', 'عند', 'أو',
]

GRADIO_CONFIG = {
    'theme': 'default',
    'share': False,
    'server_port': 7860,
}
