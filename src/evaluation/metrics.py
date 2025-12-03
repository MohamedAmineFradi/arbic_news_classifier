"""Evaluation des modèles et métriques"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> Dict[str, Any]:
    """Évaluation complète du modèle"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                if y_pred_proba.ndim > 1:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba
                
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                metrics['auc_roc'] = auc(fpr, tpr)
                metrics['fpr'] = fpr.tolist()
                metrics['tpr'] = tpr.tolist()
        except Exception as e:
            logger.warning(f"AUC non calculé: {e}")
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    return metrics


def print_evaluation_results(metrics: Dict[str, Any], model_name: str = "Modèle"):
    """Afficher résultats évaluation"""
    print("\n" + "="*60)
    print(f"Résultats évaluation {model_name}")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    print("\nMatrice de confusion:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    print("\nRapport de classification:")
    report = metrics['classification_report']
    for label, values in report.items():
        if isinstance(values, dict):
            print(f"\n{label}:")
            for metric, value in values.items():
                print(f"  {metric}: {value:.4f}")
    
    print("="*60 + "\n")


def plot_confusion_matrix(cm, class_names=['موثوقة', 'مضللة'], 
                         title='Matrice confusion', save_path=None):
    """Tracer matrice de confusion"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image sauvegardée: {save_path}")
    
    plt.show()


def plot_roc_curve(metrics_dict: Dict[str, Dict], save_path=None):
    """Tracer courbe ROC"""
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in metrics_dict.items():
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc_score = metrics.get('auc_roc', 0)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Courbe ROC', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image sauvegardée: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict], save_path=None):
    """Comparer métriques entre modèles"""
    model_names = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in model_names]
    precisions = [metrics_dict[m]['precision'] for m in model_names]
    recalls = [metrics_dict[m]['recall'] for m in model_names]
    f1_scores = [metrics_dict[m]['f1_score'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Modèle', fontsize=12)
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_title('Comparaison performance', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Image sauvegardée: {save_path}")
    
    plt.show()


def compare_models(results: Dict[str, Any]) -> pd.DataFrame:
    """Comparer modèles en tableau"""
    
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Modèle': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 0):.4f}" if 'auc_roc' in metrics else 'N/A',
        })
    
    df = pd.DataFrame(comparison_data)
    return df


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = np.abs(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print_evaluation_results(metrics, "Naive Bayes")
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm)
