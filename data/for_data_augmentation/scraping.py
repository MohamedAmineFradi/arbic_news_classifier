import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


# Liste pour stocker toutes les données extraites
data = []


# Fonction pour extraire les détails d'un article individuel
def scrape_article(url):
    """
    Extrait : titre, classification (REAL/FAKE), contenu textuel et URL
    Recherche prioritaire dans les attributs alt des images pour la classification.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extraction du titre (balise h1 principale)
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''

        # 2. Classification de véracité (REAL/FAKE)
        label = "غير محدد"  # Label par défaut
        
        # Recherche dans les attributs alt des images (méthode principale pour fatabyyano)
        images = soup.find_all('img')
        for img in images:
            if 'alt' in img.attrs:
                alt_text = img['alt']
                # Détection des mots-clés de classification
                if any(word in alt_text for word in ['زائف', 'مضلل']):
                    label = 'FAKE'
                    break
                elif any(word in alt_text for word in ['صحيح', 'حقيقة']):
                    label = 'REAL'
                    break

        # 3. Extraction du contenu principal de l'article
        content_div = soup.find('div', class_='entry-content')  # Classe standard WordPress
        if not content_div:
            # Fallback pour d'autres structures possibles
            content_div = soup.find('div', class_='post-content') or soup.find('article')
        
        text = content_div.get_text(strip=True) if content_div else ''
        
        # Filtre qualité : ignore les articles trop courts
        if len(text.split()) < 20:
            return None

        return {
            'title': title, 
            'text': text, 
            'label': label, 
            'url': url
        }

    except Exception as e:
        print(f"Erreur lors de l'extraction de {url}: {e}")
        return None


# === BOUCLE PRINCIPALE DE SCRAPING ===
base_url = "https://fatabyyano.net/news/page/"  # URL de base des pages de liste

# Scraping des 5 premières pages (configurable)
for page_num in range(1, 6): 
    print(f"--- Scraping de la page {page_num} ---")
    
    # Récupération de la page de liste
    page_url = f"{base_url}{page_num}/"
    response = requests.get(page_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    if response.status_code != 200:
        print(f"Erreur HTTP {response.status_code} pour la page {page_num}")
        continue
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extraction des liens des articles (structure h2.post-title > a)
    articles = soup.find_all('h2', class_='post-title')
    print(f"Nombre d'articles trouvés : {len(articles)}")
    
    for article in articles:
        # Extraction du lien de l'article
        link_tag = article.find('a')
        if not link_tag or not link_tag.get('href'):
            continue
            
        link = link_tag['href']
        print(f"Traitement : {link}")
        
        # Scraping de l'article individuel
        article_data = scrape_article(link)
        
        # Ne conserve que les articles avec classification détectée
        if article_data and article_data['label'] != "غير محدد":
            data.append(article_data)
            print(f"  ✓ Ajouté : {article_data['label']} ({len(article_data['text'].split())} mots)")
        else:
            print(f"  ✗ Ignoré (pas de classification)")
        
        # Pause anti-blocage (essentielle pour éviter le ban)
        time.sleep(random.uniform(1, 3))


# === SAUVEGARDE DES RÉSULTATS ===
if data:
    # Conversion en DataFrame pandas
    df = pd.DataFrame(data)
    
    # Suppression des doublons basés sur le titre
    initial_count = len(df)
    df.drop_duplicates(subset=['title'], inplace=True)
    print(f"Suppression des doublons : {initial_count} → {len(df)} articles")
    
    # Sauvegarde CSV avec encodage UTF-8 (important pour l'arabe)
    output_file = 'fatabyyano_dataset.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Statistiques finales
    label_counts = df['label'].value_counts()
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Fichier sauvegardé : {output_file}")
    print(f"Nombre total d'articles : {len(df)}")
    print("Répartition par classification :")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
else:
    print("Aucune donnée valide collectée.")
