import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

from src.utils import logger


# Dictionnaire des mots-clés pour identifier les labels de véracité des faits
FACT_LABEL_KEYWORDS = {
    'REAL': ['صحيح', 'حقيقة', 'سليم'],  # Mots-clés pour "vrai"
    'FAKE': ['زائف', 'مفبرك', 'خاطئ'],   # Mots-clés pour "faux"
    'MISLEADING': ['مضلل'],              # Mots-clés pour "trompeur"
    'PARTLY_TRUE': ['جزئيا'],            # Mots-clés pour "partiellement vrai"
}

# Mapping numérique des labels (0 pour vrai, 1 pour les autres catégories)
NUMERIC_MAP = {
    'REAL': 0,
    'FAKE': 1,
    'MISLEADING': 1,
    'PARTLY_TRUE': 1,
}


@dataclass
class ScrapeConfig:
    """Configuration du scraping avec paramètres par défaut."""
    base_list_url: str = "https://fatabyyano.net/news/page/{page}/"  # URL de base des pages de liste
    pages: int = 5                                                    # Nombre de pages à scraper
    delay_min: float = 1.0                                           # Délai minimum entre requêtes (secondes)
    delay_max: float = 3.0                                           # Délai maximum entre requêtes (secondes)
    timeout: int = 20                                                # Timeout des requêtes HTTP
    user_agents: Optional[List[str]] = None                          # Liste d'User-Agents rotatifs
    checkpoint_every: int = 0                                        # Sauvegarde checkpoint tous les N articles (0=désactivé)


def get_session() -> requests.Session:
    """Crée et configure une session requests avec un User-Agent par défaut."""
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/117.0 Safari/537.36'
    })
    return s


def pick_user_agent(config: ScrapeConfig) -> str:
    """Sélectionne un User-Agent aléatoire si configuré, sinon défaut."""
    if not config.user_agents:
        return 'Mozilla/5.0'
    return random.choice(config.user_agents)


def classify_label(soup: BeautifulSoup) -> Optional[str]:
    """
    Classe l'article en cherchant les mots-clés dans les attributs alt des images 
    et dans le texte des éléments HTML.
    """
    # 1. Recherche prioritaire dans les attributs alt des images
    for img in soup.find_all('img'):
        alt = (img.get('alt') or '').strip()
        for label, keywords in FACT_LABEL_KEYWORDS.items():
            if any(k in alt for k in keywords):
                return label

    # 2. Recherche dans les éléments textuels (span, div, p)
    text_chunks = []
    for tag in soup.find_all(['span', 'div', 'p']):
        txt = tag.get_text(separator=' ', strip=True)
        if txt:
            text_chunks.append(txt)
    joined = ' '.join(text_chunks)
    for label, keywords in FACT_LABEL_KEYWORDS.items():
        if any(k in joined for k in keywords):
            return label

    return None  # Aucun label détecté


def scrape_article(url: str, session: requests.Session, config: ScrapeConfig) -> Optional[Dict]:
    """
    Extrait les données d'un article individuel :
    - Titre (h1)
    - Contenu (div.entry-content ou fallback)
    - Label de véracité
    - URL
    Filtre les articles trop courts (<30 mots).
    """
    try:
        r = session.get(url, timeout=config.timeout)
        if r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} عند {url}")
            return None
        soup = BeautifulSoup(r.content, 'html.parser')

        # Extraction du titre
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''

        # Extraction du contenu principal
        content_div = soup.find('div', class_='entry-content')
        if not content_div:
            # Fallback si la classe n'est pas trouvée
            content_div = soup.find('div', class_='post-content')
        text = content_div.get_text(separator=' ', strip=True) if content_div else ''

        # Classification et mapping numérique
        label_raw = classify_label(soup)
        numeric_label = NUMERIC_MAP.get(label_raw) if label_raw else None

        # Filtre de qualité : ignore les textes trop courts
        if len(text.split()) < 30:
            return None

        return {
            'title': title,
            'text': text,
            'fact_label': label_raw,
            'label': numeric_label,
            'url': url
        }
    except Exception as e:
        logger.error(f"خطأ في استخراج {url}: {e}")
        return None


def scrape_listing(page: int, session: requests.Session, config: ScrapeConfig) -> List[str]:
    """
    Extrait les URLs des articles depuis une page de liste.
    Recherche les liens dans les balises h2.post-title > a.
    """
    url = config.base_list_url.format(page=page)
    try:
        r = session.get(url, timeout=config.timeout,
                        headers={'User-Agent': pick_user_agent(config)})
        if r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} صفحة {page}")
            return []
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Extraction des liens d'articles
        articles = soup.find_all('h2', class_='post-title')
        links = []
        for art in articles:
            a = art.find('a')
            if a and a.get('href'):
                links.append(a['href'])
        return links
    except Exception as e:
        logger.error(f"خطأ في الصفحة {page}: {e}")
        return []


def save_checkpoint(rows: List[Dict], out_path: Path, suffix: str):
    """Sauvegarde un checkpoint intermédiaire au format CSV."""
    df = pd.DataFrame(rows)
    cp = out_path.parent / f"{out_path.stem}_checkpoint_{suffix}.csv"
    df.to_csv(cp, index=False)
    logger.info(f"حفظ نقطة مرحلية: {cp} ({len(df)} صف)")


def run_scraper(config: ScrapeConfig, out: str):
    """
    Fonction principale du scraping :
    1. Parcourt les pages de liste
    2. Extrait les articles individuels
    3. Filtre par label de véracité
    4. Sauvegarde checkpoints + fichier final
    5. Supprime les doublons
    """
    session = get_session()
    all_rows: List[Dict] = []
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Boucle principale sur les pages
    for page in range(1, config.pages + 1):
        logger.info(f"=== صفحة {page} ===")
        links = scrape_listing(page, session, config)
        logger.info(f"عدد الروابط المكتشفة: {len(links)}")
        
        for link in links:
            art = scrape_article(link, session, config)
            # Ne garde que les articles avec label détecté
            if art and art['label'] is not None:
                all_rows.append(art)
            
            # Délai anti-blocage entre requêtes
            time.sleep(random.uniform(config.delay_min, config.delay_max))

            # Sauvegarde checkpoint périodique
            if config.checkpoint_every and len(all_rows) % config.checkpoint_every == 0:
                save_checkpoint(all_rows, out_path, f"{len(all_rows)}")

    # Post-traitement final
    df = pd.DataFrame(all_rows)
    if not df.empty:
        before = len(df)
        # Suppression des doublons basés sur titre+texte
        df.drop_duplicates(subset=['title', 'text'], inplace=True)
        logger.info(f"إزالة التكرارات: من {before} إلى {len(df)}")

    # Sauvegarde finale
    df.to_csv(out_path, index=False)
    logger.info(f"تم الحفظ النهائي: {out_path} ({len(df)} صف)؛ أعمدة: {list(df.columns)}")


def main():
    """Point d'entrée avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description="سحب مقالات فتبينوا مع التصنيف")
    parser.add_argument('--pages', type=int, default=5, 
                       help='Nombre de pages du catalogue à scraper')
    parser.add_argument('--delay_min', type=float, default=1.0, 
                       help='Délai minimum entre articles (secondes)')
    parser.add_argument('--delay_max', type=float, default=3.0, 
                       help='Délai maximum entre articles (secondes)')
    parser.add_argument('--out', type=str, default='data/processed/fatabyyano.csv', 
                       help='Fichier CSV de sortie')
    parser.add_argument('--checkpoint_every', type=int, default=0, 
                       help='Sauvegarde checkpoint tous les N articles')

    args = parser.parse_args()
    cfg = ScrapeConfig(
        pages=args.pages, 
        delay_min=args.delay_min, 
        delay_max=args.delay_max,
        checkpoint_every=args.checkpoint_every
    )

    run_scraper(cfg, args.out)


if __name__ == '__main__':
    main()
