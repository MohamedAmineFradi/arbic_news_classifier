"""Fusion du dataset AFND avec classification automatique"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

from src.utils import logger


def read_scraped_file(json_path: Path, articles_key: str = 'articles') -> List[Dict]:
    """Lire le fichier scraped_articles.json"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and articles_key in data:
            inner = data[articles_key]
            if isinstance(inner, list):
                return inner
            else:
                logger.warning(f"Clé '{articles_key}' invalide dans {json_path}")
                return []
        if isinstance(data, list):
            return data
        logger.warning(f"Format non supporté dans {json_path}")
        return []
    except Exception as e:
        logger.error(f"Échec lecture {json_path}: {e}")
        return []


def load_sources_labels(sources_json_path: str) -> Dict[str, str]:
    """Charger le fichier sources.json"""
    try:
        with open(sources_json_path, 'r', encoding='utf-8') as f:
            sources = json.load(f)
        logger.info(f"{len(sources)} sources chargées depuis {sources_json_path}")
        return sources
    except Exception as e:
        logger.error(f"Échec chargement sources: {e}")
        return {}


def ingest_afnd(afnd_dir: str, sources_labels: Optional[Dict[str, str]] = None, 
                limit: Optional[int] = None, articles_key: str = 'articles') -> pd.DataFrame:
    """Fusionner tous les articles AFND en un DataFrame"""
    base = Path(afnd_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dossier introuvable: {afnd_dir}")

    rows: List[Dict] = []
    source_dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    total_articles = 0
    for source_dir in tqdm(source_dirs, desc="Lecture sources"):
        scraped_path = source_dir / 'scraped_articles.json'
        if not scraped_path.exists():
            logger.warning(f"Pas de scraped_articles.json dans {source_dir}")
            continue
        articles = read_scraped_file(scraped_path, articles_key=articles_key)
        source_name = source_dir.name
        
        source_credibility = None
        if sources_labels and source_name in sources_labels:
            source_credibility = sources_labels[source_name]
        
        for art in articles:
            title = (art.get('title') or '').strip()
            text = (art.get('text') or '').strip()
            date = art.get('date') or art.get('publication_date') or ''

            if not text or len(text.split()) < 5:
                continue

            rows.append({
                'source': source_name,
                'title': title,
                'text': text,
                'date': date,
                'source_credibility': source_credibility
            })

            total_articles += 1
            if limit and total_articles >= limit:
                logger.info(f"Limite atteinte: {limit} articles")
                break
        if limit and total_articles >= limit:
            break

    expected_cols = ['source', 'title', 'text', 'date', 'source_credibility']
    if rows:
        df = pd.DataFrame(rows)
        label_map = {
            'credible': 0,
            'not credible': 1,
            'undecided': None
        }
        df['label'] = df['source_credibility'].map(label_map)
        
        credible_count = (df['label'] == 0).sum()
        fake_count = (df['label'] == 1).sum()
        undecided_count = df['label'].isna().sum()
        logger.info(f"Classif: {credible_count} fiables | {fake_count} faux | {undecided_count} indécis")
    else:
        df = pd.DataFrame(columns=expected_cols + ['label'])
    
    logger.info(f"Total articles agrégés: {len(df)}")
    if df.empty:
        logger.warning("Aucun article collecté. Vérifier les données sources.")
    return df


def save_output(df: pd.DataFrame, out_path: str):
    """Sauvegarder le DataFrame"""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Fichier sauvegardé: {out_file} ({len(df)} lignes)")


def main():
    """Fusion des données AFND"""
    parser = argparse.ArgumentParser(description="Fusion données AFND en CSV avec classification auto")
    parser.add_argument('--afnd_dir', type=str, default='data/AFND/Dataset', help='Dossier Dataset AFND')
    parser.add_argument('--sources_json', type=str, default='data/AFND/sources.json', help='Fichier sources.json')
    parser.add_argument('--out', type=str, default='data/processed/afnd_articles.csv', help='Fichier de sortie CSV')
    parser.add_argument('--limit', type=int, default=None, help='Limite articles (optionnel)')
    parser.add_argument('--articles_key', type=str, default='articles', help='Clé JSON pour tableau')
    parser.add_argument('--drop_unlabeled', action='store_true', help='Supprimer articles non classés')

    args = parser.parse_args()

    logger.info("Fusion données AFND...")
    
    sources_labels = None
    if os.path.exists(args.sources_json):
        sources_labels = load_sources_labels(args.sources_json)
    else:
        logger.warning(f"Fichier sources absent: {args.sources_json}. Continuer sans classification auto.")
    
    articles_df = ingest_afnd(args.afnd_dir, sources_labels=sources_labels, 
                             limit=args.limit, articles_key=args.articles_key)
    if articles_df.empty:
        logger.error("DataFrame AFND vide. Vérifier les sources.")
        return

    if args.drop_unlabeled:
        before = len(articles_df)
        articles_df = articles_df[articles_df['label'].notna()].copy()
        logger.info(f"Articles sans label supprimés: {before} → {len(articles_df)}")

    save_output(articles_df, args.out)
    logger.info("Préparation données terminée.")


if __name__ == '__main__':
    main()
