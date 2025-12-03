"""File I/O operations"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from .logging_config import logger


def save_model(model: Any, filepath: str) -> None:
    """Sauvegarder le modèle"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Modèle sauvegardé: {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise


def load_model(filepath: str) -> Any:
    """Charger le modèle"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modèle chargé: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise


def save_json(data: Dict, filepath: str) -> None:
    """Sauvegarder en JSON"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON sauvegardé: {filepath}")
    except Exception as e:
        logger.error(f"Erreur JSON: {e}")
        raise


def load_json(filepath: str) -> Dict:
    """Charger JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON chargé: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Erreur JSON: {e}")
        raise


def load_dataset(filepath: str) -> pd.DataFrame:
    """Charger le dataset"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Format non supporté: {filepath}")
        
        logger.info(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except Exception as e:
        logger.error(f"Erreur chargement dataset: {e}")
        raise
