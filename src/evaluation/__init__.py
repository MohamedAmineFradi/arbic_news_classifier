"""Model evaluation and metrics module"""

from .metrics import (
    evaluate_model, 
    print_evaluation_results,
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_metrics_comparison,
    compare_models
)

__all__ = [
    'evaluate_model',
    'print_evaluation_results',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_metrics_comparison',
    'compare_models',
]
