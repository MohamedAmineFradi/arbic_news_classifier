"""Interface Gradio pour détection fake news"""

import gradio as gr
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from src.data import ArabicTextPreprocessor
    from src.utils import load_model
    from config import CLASSES, MODELS_DIR
except ImportError as e:
    print(f"Erreur: Impossible d'importer les modules. ({e})")
    CLASSES = {0: "Fake", 1: "Real"}
    MODELS_DIR = Path("models")
    class ArabicTextPreprocessor:
        def preprocess(self, text): return text
    def load_model(path): return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    .gradio-container {
        font-family: 'Cairo', 'Segoe UI', sans-serif !important;
        direction: rtl;
        background: linear-gradient(to bottom, #f8fafc 0%, #eef2ff 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
    }
    
    .info-card {
        background: white;
        border-right: 4px solid #4f46e5;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-top: 1.5rem;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.7);
        padding: 0.8rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
"""

class FakeNewsApp:
    """Application de détection fake news"""
    
    MODEL_NAMES = {
        'nb': 'Naïve Bayes',
        'svm': 'SVM',
        'lr': 'Logistic Regression',
        'arabert': 'AraBERT (Transformer)'
    }

    FAKE_HEURISTICS = {
        'ديناصور', 'فضائي', 'فضائية', 'مخلوق غريب', 'معجزة',
        'سحري', 'سحرية', 'يوم القيامة', 'نهاية العالم', 'عاجل جدا',
        'تسريب خطير', 'كارثة عالمية', 'انفجار شمسي', 'التنبؤات',
        'الأبراج تكشف', 'يحول الماء إلى وقود'
    }
    
    def __init__(self):
        self.preprocessor = ArabicTextPreprocessor()
        self.extractor = None
        self.loaded_models: Dict[str, Any] = {}
        self.current_model_type = 'nb'
        
        self._load_feature_extractor()
        self.load_model_by_type('nb')

    def _load_feature_extractor(self):
        """Charger l'extracteur de features"""
        try:
            extractor_path = MODELS_DIR / 'feature_extractor.pkl'
            if extractor_path.exists():
                self.extractor = load_model(str(extractor_path))
                logger.info("Extracteur chargé")
            else:
                logger.warning("Extracteur manquant")
        except Exception as e:
            logger.error(f"Erreur chargement extracteur: {e}")

    def load_model_by_type(self, model_key: str) -> bool:
        """Charger un modèle en cache"""
        if model_key in self.loaded_models:
            self.current_model_type = model_key
            return True

        if model_key == 'arabert':
            try:
                logger.info("Chargement AraBERT...")
                from src.models import AraBERTFakeNewsClassifier
                model = AraBERTFakeNewsClassifier()
                self.loaded_models['arabert'] = model
                self.current_model_type = 'arabert'
                logger.info("AraBERT chargé")
                return True
            except Exception as e:
                logger.error(f"Échec AraBERT: {e}")
                return False

        try:
            model_path = MODELS_DIR / f"{model_key}_model.pkl"
            if not model_path.exists():
                logger.error(f"Modèle introuvable: {model_path}")
                return False
            
            model = load_model(str(model_path))
            self.loaded_models[model_key] = model
            self.current_model_type = model_key
            logger.info(f"Modèle {model_key} chargé")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement {model_key}: {e}")
            return False

    def _apply_heuristics(self, text: str, original_pred: int, original_proba: list) -> Tuple[int, list]:
        """Appliquer heuristiques simples"""
        lowered_text = text.lower()
        proba_diff = abs(original_proba[0] - original_proba[1])
        
        heuristic_triggered = any(tok in lowered_text for tok in self.FAKE_HEURISTICS)
        
        if heuristic_triggered and proba_diff < 0.20:
            new_proba = original_proba.copy()
            if new_proba[1] < 0.6: 
                new_proba[1] = 0.65
                new_proba[0] = 0.35
            return 1, new_proba
            
        return original_pred, original_proba

    def _generate_html_result(self, label: str, confidence: float, stats: dict) -> str:
        """Générer HTML du résultat"""
        is_trusted = (label == "موثوقة")
        
        if is_trusted:
            theme = {"icon": "✅", "color": "#059669", "bg": "linear-gradient(135deg, #d1fae5 0%, #6ee7b7 100%)", "status": "موثوقة"}
        else:
            theme = {"icon": "🚫", "color": "#dc2626", "bg": "linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%)", "status": "مضللة"}

        return f"""
        <div class="result-card" style="background: {theme['bg']}; border: 2px solid {theme['color']};">
            <div style="font-size: 4rem; margin-bottom: 0.5rem; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));">{theme['icon']}</div>
            <h2 style="color: {theme['color']}; margin: 0; font-size: 2rem; font-weight: 800;">{theme['status']}</h2>
            
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 1.1rem; color: #4b5563; margin-bottom: 0.5rem;">نسبة الثقة</div>
                <div style="font-size: 3rem; font-weight: bold; color: {theme['color']}; line-height: 1;">{confidence:.1%}</div>
            </div>

            <div style="background: rgba(255,255,255,0.6); height: 10px; border-radius: 5px; overflow: hidden; margin: 1rem auto; width: 80%;">
                <div style="width: {confidence*100}%; height: 100%; background: {theme['color']}; transition: width 1s ease;"></div>
            </div>

            <div class="stat-grid">
                <div class="stat-box">
                    <small>كلمات</small><br><strong>{stats['words']}</strong>
                </div>
                <div class="stat-box">
                    <small>أحرف</small><br><strong>{stats['chars']}</strong>
                </div>
                <div class="stat-box">
                    <small>النموذج</small><br><strong style="text-transform: uppercase;">{stats['model']}</strong>
                </div>
            </div>
        </div>
        """

    def predict(self, text: str, model_display_name: str) -> Tuple[str, Optional[Dict]]:
        """Prédiction principale"""
        
        if not text or not text.strip():
            return """<div style="text-align:center; padding:2rem; background:#fff3cd; color:#856404; border-radius:1rem;">⚠️ Veuillez entrer un texte</div>""", None

        model_key = next((k for k, v in self.MODEL_NAMES.items() if v == model_display_name), 'nb')

        if model_key not in self.loaded_models:
            success = self.load_model_by_type(model_key)
            if not success:
                return f"""<div style="text-align:center; padding:2rem; background:#f8d7da; color:#721c24; border-radius:1rem;">❌ Impossible de charger {model_key}</div>""", None

        model = self.loaded_models[model_key]

        try:
            processed_text = self.preprocessor.preprocess(text)
            
            if model_key == 'arabert':
                preds, probs = model.predict([processed_text])
                prediction = int(preds[0])
                proba = probs[0]
            else:
                if self.extractor is None:
                     return "Erreur: Extracteur non chargé", None
                X = self.extractor.transform([processed_text])
                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0]

            final_pred, final_proba = self._apply_heuristics(text, int(prediction), proba)

            label_name = CLASSES[final_pred]
            confidence = final_proba[final_pred]
            
            stats = {
                'words': len(text.split()),
                'chars': len(text),
                'model': model_key
            }

            html_output = self._generate_html_result(label_name, confidence, stats)
            
            proba_dict = {
                f"{CLASSES[0]} (موثوقة)": float(final_proba[0]),
                f"{CLASSES[1]} (مضللة)": float(final_proba[1])
            }
            
            return html_output, proba_dict

        except Exception as e:
            logger.error(f"Erreur prédiction: {e}", exc_info=True)
            return f"""<div style="text-align:center; padding:2rem; background:#f8d7da; color:#721c24;">Erreur: {str(e)}</div>""", None


def create_interface():
    app = FakeNewsApp()
    
    examples_list = [
        ["أعلنت وزارة الصحة رسمياً عن انخفاض معدلات الإصابة بنسبة 20% هذا الأسبوع."],
        ["عاجل: كائنات فضائية تهبط في الأهرامات والحكومة تتستر على الأمر!"],
        ["شاهد بالفيديو: عشبة سحرية تشفي من جميع الأمراض المستعصية في يوم واحد."]
    ]

    with gr.Blocks(title="Détecteur Fake News") as demo:
        gr.HTML(CSS_STYLES)
        
        gr.HTML("""
        <div class="main-header">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍 المحقق الذكي للأخبار</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">تحليل مصداقية النصوص العربية باستخدام الذكاء الاصطناعي</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="info-card">
                    <h3 style="color: #4f46e5; margin-top:0;">📝 النص المراد تحليله</h3>
                    <p style="color: #666; font-size: 0.9rem;">قم بنسخ ولصق عنوان الخبر أو متن المقال هنا.</p>
                </div>
                """)
                
                input_text = gr.Textbox(
                    label="", 
                    placeholder="ضع النص هنا...", 
                    lines=6, 
                    elem_id="input_box"
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(app.MODEL_NAMES.values()),
                        value=app.MODEL_NAMES['nb'],
                        label="نموذج التحليل",
                        interactive=True,
                        scale=2
                    )
                    
                    analyze_btn = gr.Button("🔍 تحليل", variant="primary", scale=1)
                    clear_btn = gr.ClearButton([input_text], value="مسح", scale=0)

            with gr.Column(scale=2):
                gr.HTML("""
                <div class="info-card">
                    <h3 style="color: #4f46e5; margin-top:0;">📊 تقرير المصداقية</h3>
                </div>
                """)
                output_html = gr.HTML(label="النتيجة")
                output_chart = gr.Label(label="التفاصيل", num_top_classes=2)

        gr.Examples(
            examples=examples_list,
            inputs=input_text,
            label="جرب هذه الأمثلة"
        )
        
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        ⚠️ <b>تنبيه:</b> هذه الأداة مساعدة وتعتمد على خوارزميات احتمالية. يرجى دائماً التحقق من المصادر الرسمية.
        </div>
        """)

        analyze_btn.click(
            fn=app.predict,
            inputs=[input_text, model_dropdown],
            outputs=[output_html, output_chart]
        )
        
        input_text.submit(
            fn=app.predict,
            inputs=[input_text, model_dropdown],
            outputs=[output_html, output_chart]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )