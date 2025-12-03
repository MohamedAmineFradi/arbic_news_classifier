"""
Interface Gradio modulaire - Detective Conan Fake News Detector
Application décomposée en modules séparés avec thème Detective Conan
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Tuple, Optional

# Import des modules personnalisés
from .theme_conan import (
    CSS_CONAN_THEME,
    get_header_html,
    get_input_card_html,
    get_result_card_html,
    get_footer_html,
    CONAN_EXAMPLES
)
from .model_handler import ModelHandler
from .result_generator import (
    generate_result_html,
    generate_error_html,
    generate_loading_html
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Essayer d'importer la config
try:
    import sys
    ROOT_DIR = Path(__file__).parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from config import CLASSES
except ImportError:
    logger.warning("Config non trouvée, utilisation des valeurs par défaut")
    CLASSES = {0: "موثوقة", 1: "مضللة"}


class ConanFakeNewsDetector:
    """
    Détecteur de Fake News style Detective Conan
    
    "La vérité est toujours une !" - Detective Conan
    """
    
    def __init__(self):
        logger.info("🕵️ Initialisation du Détecteur Conan...")
        self.model_handler = ModelHandler()
        logger.info("✅ Détecteur Conan prêt!")
    
    def analyze_text(
        self,
        text: str,
        model_display_name: str
    ) -> Tuple[str, Optional[dict]]:
        """
        Analyser un texte pour détecter les fake news
        
        Args:
            text: Texte à analyser
            model_display_name: Nom du modèle affiché
            
        Returns:
            Tuple (HTML résultat, probabilities dict)
        """
        
        # Validation du texte
        if not text or not text.strip():
            error_html = generate_error_html(
                "⚠️ من فضلك أدخل نصاً للتحليل",
                error_type="warning"
            )
            return error_html, None
        
        if len(text.strip()) < 10:
            error_html = generate_error_html(
                "⚠️ النص قصير جداً. يحتاج المحقق كونان إلى مزيد من التفاصيل!",
                error_type="warning"
            )
            return error_html, None
        
        # Trouver la clé du modèle
        model_key = self._get_model_key(model_display_name)
        
        try:
            # Effectuer la prédiction
            logger.info(f"🔍 Analyse avec {model_display_name}...")
            
            prediction, probabilities, metadata = self.model_handler.predict(
                text=text,
                model_key=model_key
            )
            
            # Préparer les résultats
            is_reliable = (prediction == 0)
            confidence = probabilities[prediction]
            
            stats = {
                'words': metadata['word_count'],
                'chars': metadata['text_length'],
                'model': model_key.upper()
            }
            
            # Générer le HTML
            result_html = generate_result_html(
                is_reliable=is_reliable,
                confidence=confidence,
                stats=stats,
                adjusted_by_heuristics=metadata.get('adjusted_by_heuristics', False)
            )
            
            # Préparer les probabilités pour l'affichage
            proba_dict = {
                f"✅ {CLASSES[0]} (Fiable)": float(probabilities[0]),
                f"🚫 {CLASSES[1]} (Fake)": float(probabilities[1])
            }
            
            logger.info(f"✅ Analyse terminée: {CLASSES[prediction]} ({confidence:.1%})")
            
            return result_html, proba_dict
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse: {e}", exc_info=True)
            error_html = generate_error_html(
                f"❌ خطأ في التحليل: {str(e)}",
                error_type="error"
            )
            return error_html, None
    
    def _get_model_key(self, display_name: str) -> str:
        """Convertir le nom affiché en clé de modèle"""
        for key, name in self.model_handler.MODEL_NAMES.items():
            if name == display_name:
                return key
        return 'nb'  # Par défaut
    
    def get_available_models(self):
        """Obtenir la liste des modèles disponibles"""
        return self.model_handler.get_available_models()


def create_conan_interface() -> gr.Blocks:
    """
    Créer l'interface Gradio avec le thème Detective Conan
    
    Returns:
        Interface Gradio configurée
    """
    
    logger.info("🎨 Création de l'interface Detective Conan...")
    
    # Initialiser le détecteur
    detector = ConanFakeNewsDetector()
    available_models = detector.get_available_models()
    
    # Si aucun modèle disponible, utiliser liste par défaut
    if not available_models:
        available_models = list(ModelHandler.MODEL_NAMES.values())
    
    # Créer l'interface
    with gr.Blocks(
        title="المحقق كونان للأخبار | Detective Conan News"
    ) as demo:
        
        # Styles CSS
        gr.HTML(CSS_CONAN_THEME)
        
        # En-tête Detective Conan
        gr.HTML(get_header_html())
        
        # Layout principal
        with gr.Row():
            # Colonne gauche - Input
            with gr.Column(scale=3):
                gr.HTML(get_input_card_html())
                
                input_text = gr.Textbox(
                    label="",
                    placeholder="🔍 أدخل النص أو الخبر هنا...\n\nمثال: أعلنت وزارة الصحة عن...",
                    lines=8,
                    elem_id="input_box"
                )
                
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0] if available_models else "Naïve Bayes",
                        label="🤖 اختر نموذج التحليل",
                        interactive=True,
                        scale=3
                    )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 ابدأ التحقيق!",
                        variant="primary",
                        scale=2
                    )
                    clear_btn = gr.ClearButton(
                        components=[input_text],
                        value="🗑️ مسح",
                        scale=1
                    )
            
            # Colonne droite - Output
            with gr.Column(scale=2):
                gr.HTML(get_result_card_html())
                
                result_output = gr.HTML(
                    value="""
                    <div style="text-align:center; padding:3rem; color:#94a3b8;">
                        <div style="font-size:4rem; margin-bottom:1rem;">🕵️‍♂️</div>
                        <div style="font-size:1.2rem; font-weight:600;">
                            في انتظار القضية...
                        </div>
                        <div style="font-size:0.9rem; margin-top:0.5rem;">
                            المحقق كونان جاهز للتحليل!
                        </div>
                    </div>
                    """,
                    label=""
                )
                
                probability_output = gr.Label(
                    label="📊 احتمالات التصنيف",
                    num_top_classes=2
                )
        
        # Section des exemples style cas d'enquête
        gr.Markdown("### 📁 قضايا للتجربة - حالات تحقيق نموذجية")
        gr.Examples(
            examples=CONAN_EXAMPLES,
            inputs=input_text,
            label="",
            examples_per_page=5
        )
        
        # Footer
        gr.HTML(get_footer_html())
        
        # Événements
        analyze_btn.click(
            fn=detector.analyze_text,
            inputs=[input_text, model_selector],
            outputs=[result_output, probability_output]
        )
        
        input_text.submit(
            fn=detector.analyze_text,
            inputs=[input_text, model_selector],
            outputs=[result_output, probability_output]
        )
    
    logger.info("✅ Interface Detective Conan créée avec succès!")
    return demo


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """
    Lancer l'application Detective Conan
    
    Args:
        server_name: Nom du serveur
        server_port: Port du serveur
        share: Créer un lien public Gradio
        debug: Mode debug
    """
    
    logger.info("=" * 60)
    logger.info("🕵️ DETECTIVE CONAN FAKE NEWS DETECTOR 🔍")
    logger.info("=" * 60)
    
    demo = create_conan_interface()
    
    logger.info(f"🚀 Lancement sur {server_name}:{server_port}")
    logger.info("📝 La vérité est toujours une! - Detective Conan")
    
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug
    )


if __name__ == "__main__":
    launch_app()
