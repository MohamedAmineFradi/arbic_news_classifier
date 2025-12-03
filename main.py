"""Programme principal pour l'entraînement et l'évaluation des modèles"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import ArabicTextPreprocessor, ingest_afnd, load_sources_labels
from src.features import FeatureExtractor, TextFeatureEngineering
from src.models import FakeNewsDetector, train_multiple_models
from src.evaluation import (
    evaluate_model, print_evaluation_results, 
    plot_confusion_matrix, plot_metrics_comparison, compare_models
)
from src.utils import (
    load_dataset, save_model, split_data, 
    get_label_distribution, logger
)
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIG, CLASSES
)


def load_and_prepare_data(data_path: str):
    """Charger et préparer les données"""
    logger.info("Chargement des données...")
    df = load_dataset(data_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Les données doivent contenir les colonnes 'text' et 'label'")
    
    df = df.dropna(subset=['text', 'label'])
    logger.info(f"Nombre d'échantillons: {len(df)}")
    get_label_distribution(df['label'].values)
    
    return df


def preprocess_texts(texts, preprocessor=None):
    """Traiter les textes"""
    if preprocessor is None:
        preprocessor = ArabicTextPreprocessor()
    
    logger.info("Traitement des textes...")
    processed_texts = preprocessor.preprocess_batch(texts.tolist())
    return processed_texts


def extract_features(texts, method='tfidf', max_features=5000):
    """Extraire les caractéristiques"""
    logger.info(f"Extraction des caractéristiques avec {method}...")
    extractor = FeatureExtractor(method=method, max_features=max_features)
    X = extractor.fit_transform(texts)
    return X, extractor


def train_and_evaluate(data_path: str, model_types=['nb', 'svm', 'rf']):
    """Entraîner et évaluer les modèles"""
    df = load_and_prepare_data(data_path)
    processed_texts = preprocess_texts(df['text'])
    
    processed_df = pd.DataFrame({
        'text': df['text'].values,
        'processed_text': processed_texts,
        'label': df['label'].values
    })
    processed_path = PROCESSED_DATA_DIR / 'processed_data.csv'
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"Données traitées sauvegardées: {processed_path}")
    
    X, extractor = extract_features(
        processed_texts, 
        method='tfidf',
        max_features=MODEL_CONFIG['max_features']
    )
    y = df['label'].values
    
    extractor_path = MODELS_DIR / 'feature_extractor.pkl'
    save_model(extractor, extractor_path)
    
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )
    
    logger.info("\n" + "="*60)
    logger.info("Début de l'entraînement des modèles...")
    logger.info("="*60)
    
    results = {}
    
    for model_type in model_types:
        logger.info(f"\nEntraînement du modèle {model_type}...")
        
        try:
            detector = FakeNewsDetector(model_type=model_type)
            detector.train(X_train, y_train)
            
            y_pred = detector.predict(X_test)
            y_pred_proba = detector.predict_proba(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            print_evaluation_results(metrics, model_type.upper())
            
            model_path = MODELS_DIR / f'{model_type}_model.pkl'
            detector.save(model_path)
            
            results[model_type] = {
                'model': detector,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de {model_type}: {e}")
            continue
    
    if len(results) > 1:
        logger.info("\n" + "="*60)
        logger.info("Comparaison des modèles")
        logger.info("="*60)
        
        comparison_df = compare_models(results)
        print("\n", comparison_df)
        
        metrics_dict = {name: res['metrics'] for name, res in results.items()}
        plot_metrics_comparison(metrics_dict, save_path=MODELS_DIR / 'comparison.png')
    
    return results


def predict_single_text(text: str, model_path: str, extractor_path: str):
    """Prédire pour un texte unique"""
    from src.utils import load_model
    
    detector = FakeNewsDetector()
    detector.load(model_path)
    extractor = load_model(extractor_path)
    
    preprocessor = ArabicTextPreprocessor()
    processed_text = preprocessor.preprocess(text)
    X = extractor.transform([processed_text])
    
    prediction = detector.predict(X)[0]
    proba = detector.predict_proba(X)[0]
    
    result = {
        'label': prediction,
        'label_name': CLASSES[prediction],
        'confidence': float(proba[prediction]),
        'probabilities': {
            CLASSES[0]: float(proba[0]),
            CLASSES[1]: float(proba[1])
        }
    }
    
    return result


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Détection de fake news en arabe')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'ingest_afnd'],
                        help='Mode: train, predict, ou ingest_afnd')
    parser.add_argument('--afnd_dir', type=str, default='data/AFND/Dataset',
                        help='Dossier Dataset AFND')
    parser.add_argument('--afnd_limit', type=int, default=None,
                        help='Limite d\'articles AFND (optionnel)')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='Fichier de labels externe (CSV/JSON)')
    parser.add_argument('--drop_unlabeled', action='store_true',
                        help='Supprimer les articles non labellisés')
    
    parser.add_argument('--data', type=str, default='data/processed/afnd_balanced.csv',
                       help='Chemin du fichier de données')
    
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['nb', 'svm', 'rf'],
                       help='Types de modèles à entraîner')
    
    parser.add_argument('--text', type=str,
                       help='Texte à prédire (mode predict)')
    
    parser.add_argument('--model_path', type=str,
                       default='models/saved_models/nb_model.pkl',
                       help='Chemin du modèle sauvegardé')
    
    parser.add_argument('--extractor_path', type=str,
                       default='models/saved_models/feature_extractor.pkl',
                       help='Chemin de l\'extracteur')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info("Démarrage de l'entraînement...")
        results = train_and_evaluate(args.data, args.models)
        logger.info("Entraînement terminé avec succès!")
        
    elif args.mode == 'predict':
        if not args.text:
            logger.error("Veuillez fournir un texte avec --text")
            return
        
        logger.info("Prédiction en cours...")
        result = predict_single_text(args.text, args.model_path, args.extractor_path)
        
        print("\n" + "="*60)
        print("Résultat de la prédiction")
        print("="*60)
        print(f"Classification: {result['label_name']}")
        print(f"Confiance: {result['confidence']:.2%}")
        print("\nProbabilités:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print("="*60)

    elif args.mode == 'ingest_afnd':
        logger.info("Fusion des données AFND...")
        from pathlib import Path
        sources_json = Path(args.afnd_dir).parent / 'sources.json'
        sources_labels = load_sources_labels(str(sources_json)) if sources_json.exists() else None
        articles_df = ingest_afnd(args.afnd_dir, sources_labels=sources_labels, limit=args.afnd_limit)
        if args.drop_unlabeled:
            before = len(articles_df)
            articles_df = articles_df[articles_df['label'].notna()].copy()
            logger.info(f"Articles non labellisés supprimés: de {before} à {len(articles_df)}")
        out_path = PROCESSED_DATA_DIR / 'afnd_articles.csv'
        articles_df.to_csv(out_path, index=False)
        logger.info(f"Fichier AFND fusionné sauvegardé: {out_path}")
        print("\n" + "="*60)
        print("Statistiques AFND:")
        print("="*60)
        print(f"Nombre de lignes: {len(articles_df)}")
        if 'label' in articles_df.columns:
            label_counts = articles_df['label'].value_counts(dropna=False)
            print("\nDistribution des labels:")
            print(label_counts)
        print("="*60)


if __name__ == "__main__":
    main()
