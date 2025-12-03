(venv) bl4z@my-latptop:~/Downloads/projet_nlp_fake_news_arabe$ source /home/bl4z/Downloads/projet_nlp_fake_news_arabe/venv/bin/activate && python main.py --mode train --data data/processed/afnd_balanced.csv --models nb svm rf lr gb
2025-12-02 00:32:55,770 - src.utils.logging_config - INFO - Démarrage de l'entraînement...
2025-12-02 00:32:55,770 - src.utils.logging_config - INFO - Chargement des données...
2025-12-02 00:32:56,067 - src.utils.logging_config - INFO - Dataset chargé: 7074 lignes, 6 colonnes
2025-12-02 00:32:56,068 - src.utils.logging_config - INFO - Nombre d'échantillons: 7074

Distribution des labels:
  0.0: 3537 (50.00%)
  1.0: 3537 (50.00%)
2025-12-02 00:32:56,400 - src.utils.logging_config - INFO - Traitement des textes...
Traitement: 100%|████████████████████████████████████████████████████| 7074/7074 [05:05<00:00, 23.15it/s]
2025-12-02 00:38:02,371 - src.utils.logging_config - INFO - Données traitées sauvegardées: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/data/processed/processed_data.csv
2025-12-02 00:38:02,371 - src.utils.logging_config - INFO - Extraction des caractéristiques avec tfidf...
2025-12-02 00:38:02,371 - src.features.feature_extraction - INFO - Extraction features avec tfidf...
2025-12-02 00:38:07,409 - src.features.feature_extraction - INFO - Forme matrice: (7074, 5000)
2025-12-02 00:38:07,421 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/feature_extractor.pkl
2025-12-02 00:38:07,428 - src.utils.logging_config - INFO - Taille train: 5659
2025-12-02 00:38:07,429 - src.utils.logging_config - INFO - Taille test: 1415
2025-12-02 00:38:07,429 - src.utils.logging_config - INFO - 
============================================================
2025-12-02 00:38:07,429 - src.utils.logging_config - INFO - Début de l'entraînement des modèles...
2025-12-02 00:38:07,429 - src.utils.logging_config - INFO - ============================================================
2025-12-02 00:38:07,429 - src.utils.logging_config - INFO - 
Entraînement du modèle nb...
2025-12-02 00:38:07,429 - src.models.classical_models - INFO - Modèle nb créé
2025-12-02 00:38:07,429 - src.models.classical_models - INFO - Entraînement du modèle nb...
2025-12-02 00:38:07,433 - src.models.classical_models - INFO - Entraînement terminé

============================================================
Résultats évaluation NB
============================================================

Accuracy: 0.9350
Precision: 0.9350
Recall: 0.9350
F1-Score: 0.9350
AUC-ROC: 0.9796

Matrice de confusion:
[[661  47]
 [ 45 662]]

Rapport de classification:

0.0:
  precision: 0.9363
  recall: 0.9336
  f1-score: 0.9349
  support: 708.0000

1.0:
  precision: 0.9337
  recall: 0.9364
  f1-score: 0.9350
  support: 707.0000

macro avg:
  precision: 0.9350
  recall: 0.9350
  f1-score: 0.9350
  support: 1415.0000

weighted avg:
  precision: 0.9350
  recall: 0.9350
  f1-score: 0.9350
  support: 1415.0000
============================================================

2025-12-02 00:38:07,446 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/nb_model.pkl
2025-12-02 00:38:07,446 - src.utils.logging_config - INFO - 
Entraînement du modèle svm...
2025-12-02 00:38:07,446 - src.models.classical_models - INFO - Modèle svm créé
2025-12-02 00:38:07,446 - src.models.classical_models - INFO - Entraînement du modèle svm...
2025-12-02 00:38:40,790 - src.models.classical_models - INFO - Calibration des probabilités...
2025-12-02 00:39:35,752 - src.models.classical_models - INFO - Calibration terminée
2025-12-02 00:39:35,752 - src.models.classical_models - INFO - Entraînement terminé

============================================================
Résultats évaluation SVM
============================================================

Accuracy: 0.9781
Precision: 0.9781
Recall: 0.9781
F1-Score: 0.9781
AUC-ROC: 0.9981

Matrice de confusion:
[[692  16]
 [ 15 692]]

Rapport de classification:

0.0:
  precision: 0.9788
  recall: 0.9774
  f1-score: 0.9781
  support: 708.0000

1.0:
  precision: 0.9774
  recall: 0.9788
  f1-score: 0.9781
  support: 707.0000

macro avg:
  precision: 0.9781
  recall: 0.9781
  f1-score: 0.9781
  support: 1415.0000

weighted avg:
  precision: 0.9781
  recall: 0.9781
  f1-score: 0.9781
  support: 1415.0000
============================================================

2025-12-02 00:39:42,047 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/svm_model.pkl
2025-12-02 00:39:42,047 - src.utils.logging_config - INFO - 
Entraînement du modèle rf...
2025-12-02 00:39:42,047 - src.models.classical_models - INFO - Modèle rf créé
2025-12-02 00:39:42,047 - src.models.classical_models - INFO - Entraînement du modèle rf...
2025-12-02 00:39:44,155 - src.models.classical_models - INFO - Entraînement terminé

============================================================
Résultats évaluation RF
============================================================

Accuracy: 0.9795
Precision: 0.9796
Recall: 0.9795
F1-Score: 0.9795
AUC-ROC: 0.9979

Matrice de confusion:
[[697  11]
 [ 18 689]]

Rapport de classification:

0.0:
  precision: 0.9748
  recall: 0.9845
  f1-score: 0.9796
  support: 708.0000

1.0:
  precision: 0.9843
  recall: 0.9745
  f1-score: 0.9794
  support: 707.0000

macro avg:
  precision: 0.9796
  recall: 0.9795
  f1-score: 0.9795
  support: 1415.0000

weighted avg:
  precision: 0.9796
  recall: 0.9795
  f1-score: 0.9795
  support: 1415.0000
============================================================

2025-12-02 00:39:44,298 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/rf_model.pkl
2025-12-02 00:39:44,298 - src.utils.logging_config - INFO - 
Entraînement du modèle lr...
2025-12-02 00:39:44,298 - src.models.classical_models - INFO - Modèle lr créé
2025-12-02 00:39:44,298 - src.models.classical_models - INFO - Entraînement du modèle lr...
2025-12-02 00:39:44,335 - src.models.classical_models - INFO - Calibration des probabilités...
2025-12-02 00:39:44,430 - src.models.classical_models - INFO - Calibration terminée
2025-12-02 00:39:44,430 - src.models.classical_models - INFO - Entraînement terminé

============================================================
Résultats évaluation LR
============================================================

Accuracy: 0.9647
Precision: 0.9647
Recall: 0.9647
F1-Score: 0.9647
AUC-ROC: 0.9945

Matrice de confusion:
[[682  26]
 [ 24 683]]

Rapport de classification:

0.0:
  precision: 0.9660
  recall: 0.9633
  f1-score: 0.9646
  support: 708.0000

1.0:
  precision: 0.9633
  recall: 0.9661
  f1-score: 0.9647
  support: 707.0000

macro avg:
  precision: 0.9647
  recall: 0.9647
  f1-score: 0.9647
  support: 1415.0000

weighted avg:
  precision: 0.9647
  recall: 0.9647
  f1-score: 0.9647
  support: 1415.0000
============================================================

2025-12-02 00:39:44,450 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/lr_model.pkl
2025-12-02 00:39:44,450 - src.utils.logging_config - INFO - 
Entraînement du modèle gb...
2025-12-02 00:39:44,451 - src.models.classical_models - INFO - Modèle gb créé
2025-12-02 00:39:44,451 - src.models.classical_models - INFO - Entraînement du modèle gb...
2025-12-02 00:41:06,530 - src.models.classical_models - INFO - Entraînement terminé

============================================================
Résultats évaluation GB
============================================================

Accuracy: 0.9625
Precision: 0.9646
Recall: 0.9625
F1-Score: 0.9625
AUC-ROC: 0.9966

Matrice de confusion:
[[705   3]
 [ 50 657]]

Rapport de classification:

0.0:
  precision: 0.9338
  recall: 0.9958
  f1-score: 0.9638
  support: 708.0000

1.0:
  precision: 0.9955
  recall: 0.9293
  f1-score: 0.9612
  support: 707.0000

macro avg:
  precision: 0.9646
  recall: 0.9625
  f1-score: 0.9625
  support: 1415.0000

weighted avg:
  precision: 0.9646
  recall: 0.9625
  f1-score: 0.9625
  support: 1415.0000
============================================================

2025-12-02 00:41:06,631 - src.utils.logging_config - INFO - Modèle sauvegardé: /home/bl4z/Downloads/projet_nlp_fake_news_arabe/models/saved_models/gb_model.pkl
2025-12-02 00:41:06,631 - src.utils.logging_config - INFO - 
============================================================
2025-12-02 00:41:06,631 - src.utils.logging_config - INFO - Comparaison des modèles