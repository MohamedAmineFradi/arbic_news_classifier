# مشروع كشف الأخبار الكاذبة باللغة العربية
# Arabic Fake News Detection Project

## 📋 نظرة عامة | Overview

مشروع للكشف عن الأخبار الكاذبة والشائعات في المقالات الإخبارية العربية باستخدام تقنيات معالجة اللغة الطبيعية والتعلم الآلي.

A project to detect fake news and rumors in Arabic news articles using Natural Language Processing (NLP) and Machine Learning techniques.

## ✨ الميزات | Features

- ✅ تصنيف المقالات الإخبارية إلى "موثوقة" أو "مضللة"
- ✅ معالجة النصوص العربية (تنظيف، إزالة الكلمات المشتركة، stemming)
- ✅ استخراج الخصائص باستخدام TF-IDF و Word2Vec
- ✅ نماذج تعلم آلي متعددة (Naive Bayes, SVM, Random Forest, Neural Networks)
- ✅ واجهة مستخدم سهلة باستخدام Gradio
- ✅ تقييم شامل للنماذج

## 🛠️ المتطلبات | Requirements

```bash
pip install -r requirements.txt
```

### المكتبات الأساسية:
- pandas
- numpy
- scikit-learn
- nltk
- CAMeL-Tools (معالجة اللغة العربية)
- transformers (AraBERT)
- torch
- gradio
- matplotlib
- seaborn

## 📁 هيكل المشروع | Project Structure

```
projet_nlp_fake_news_arabe/
├── data/                       # مجلد البيانات
│   ├── raw/                    # البيانات الخام
│   ├── processed/              # البيانات المعالجة
│   └── sample_data.csv         # بيانات تجريبية
├── models/                     # النماذج المدربة
│   └── saved_models/           # النماذج المحفوظة
├── notebooks/                  # دفاتر Jupyter للتحليل
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/                        # الكود المصدري
│   ├── preprocessing.py        # معالجة النصوص
│   ├── feature_extraction.py   # استخراج الخصائص
│   ├── model_training.py       # تدريب النماذج
│   ├── model_evaluation.py     # تقييم النماذج
│   └── utils.py                # وظائف مساعدة
├── app/                        # واجهة المستخدم
│   └── gradio_app.py           # تطبيق Gradio
├── requirements.txt            # المتطلبات
├── config.py                   # الإعدادات
└── main.py                     # نقطة البداية

```

## 🚀 البدء السريع | Quick Start

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. تحميل بيانات NLTK العربية
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. تشغيل التطبيق
```bash
python main.py
```

### 4. فتح واجهة المستخدم
```bash
python app/gradio_app.py
```

## 📊 مصادر البيانات | Data Sources

يمكنك استخدام مجموعات البيانات التالية:
- **Arabic Fake News Dataset (AFND)**
- **LIAR-PLUS Dataset (Arabic translation)**
- **ANT-Arabic dataset**

## 🧪 استخدام النماذج | Model Usage

```python
from src.model_training import FakeNewsDetector

# تحميل النموذج
detector = FakeNewsDetector()
detector.load_model('models/saved_models/best_model.pkl')

# التنبؤ
text = "هذا نص المقالة الإخبارية..."
prediction = detector.predict(text)
print(f"التصنيف: {prediction}")  # موثوقة أو مضللة
```
## 📈 النتائج | Results

تم اختبار النماذج التالية وحصلنا على مقاييس الأداء الموضحة أدناه:

| Model               | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|---------------------|----------|-----------|---------|----------|---------|
| Naive Bayes         | 93.50%   | 93.50%    | 93.50%  | 93.50%   | 97.96%  |
| **SVM**             | **97.81%** | **97.81%** | **97.81%** | **97.81%** | **99.81%** |
| **Random Forest**   | **97.95%** | **97.96%** | **97.95%** | **97.95%** | 99.79%  |
| Logistic Regression | 96.47%   | 96.47%    | 96.47%  | 96.47%   | 99.45%  |
| Gradient Boosting   | 96.25%   | 96.46%    | 96.25%  | 96.25%   | 99.66%  |

**ملاحظة**: حقق نموذج Random Forest أفضل دقة (Accuracy) بينما حقق نموذج SVM أفضل منطقة تحت المنحنى (AUC-ROC).

## 🤝 المساهمة | Contributing

نرحب بالمساهمات! يرجى فتح issue أو pull request.


## 👨‍💻 المطور | Developer

مشروع تعليمي لمعالجة اللغة الطبيعية
