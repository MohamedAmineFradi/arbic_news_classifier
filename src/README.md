# Source Code Structure

This directory contains the source code organized into modular packages:

## Directory Structure

```
src/
├── __init__.py
├── data/               # Data processing and preprocessing
│   ├── __init__.py
│   ├── preprocessing.py      # Arabic text preprocessing
│   ├── ingest_afnd.py        # AFND dataset ingestion
│   └── balance_dataset.py    # Dataset balancing
│
├── features/           # Feature extraction
│   ├── __init__.py
│   └── feature_extraction.py # TF-IDF and feature engineering
│
├── models/            # Machine learning models
│   ├── __init__.py
│   ├── classical_models.py   # Traditional ML models
│   └── arabert_detector.py   # AraBERT transformer model
│
├── evaluation/        # Model evaluation and metrics
│   ├── __init__.py
│   └── metrics.py            # Evaluation functions and visualizations
│
└── utils/             # Utility functions
    ├── __init__.py
    ├── logging_config.py     # Logging configuration
    ├── file_io.py            # File I/O operations
    └── data_utils.py         # Data manipulation utilities
```

## Module Descriptions

### `data/` - Data Processing
- **preprocessing.py**: Arabic text preprocessing (normalization, tokenization, stemming)
- **ingest_afnd.py**: AFND dataset ingestion and labeling
- **balance_dataset.py**: Dataset balancing utilities

### `features/` - Feature Extraction
- **feature_extraction.py**: TF-IDF vectorization and statistical feature engineering

### `models/` - Machine Learning Models
- **classical_models.py**: Traditional ML models (Naive Bayes, SVM, Random Forest, etc.)
- **arabert_detector.py**: AraBERT transformer-based classifier

### `evaluation/` - Model Evaluation
- **metrics.py**: Evaluation metrics, confusion matrices, ROC curves, and visualizations

### `utils/` - Utilities
- **logging_config.py**: Centralized logging configuration
- **file_io.py**: Model and data persistence
- **data_utils.py**: Data manipulation and statistics

## Usage Examples

### Import from data module
```python
from src.data import ArabicTextPreprocessor, ingest_afnd, balance_dataset
```

### Import from features module
```python
from src.features import FeatureExtractor, TextFeatureEngineering
```

### Import from models module
```python
from src.models import FakeNewsDetector, AraBERTFakeNewsClassifier
```

### Import from evaluation module
```python
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
```

### Import from utils module
```python
from src.utils import logger, save_model, load_model, split_data
```

## Benefits of This Structure

1. **Modularity**: Each module has a clear, single responsibility
2. **Maintainability**: Easier to locate and update specific functionality
3. **Testability**: Isolated modules are easier to test
4. **Scalability**: Easy to add new features or models
5. **Collaboration**: Multiple developers can work on different modules
6. **Reusability**: Modules can be imported and reused across projects
