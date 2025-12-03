"""Générateur de résultats HTML pour l'interface Gradio"""

from typing import Dict, Any
from .theme_conan import CONAN_COLORS

def generate_result_html(
    is_reliable: bool,
    confidence: float,
    stats: Dict[str, Any],
    adjusted_by_heuristics: bool = False
) -> str:
    """
    Générer le HTML du résultat style Detective Conan
    
    Args:
        is_reliable: True si l'article est fiable
        confidence: Score de confiance (0-1)
        stats: Statistiques (words, chars, model, etc.)
        adjusted_by_heuristics: Si ajusté par les heuristiques Conan
        
    Returns:
        HTML formaté
    """
    
    if is_reliable:
        theme = {
            'icon': '✅',
            'secondary_icon': '🔍✓',
            'color': CONAN_COLORS['success'],
            'bg_class': 'truth-revealed',
            'verdict': 'الحقيقة!',
            'verdict_en': 'VÉRITÉ',
            'message': 'هذا النص يبدو موثوقاً',
            'conan_quote': '"الحقائق لا تخطئ، الناس هم من يخطئون" - كونان'
        }
    else:
        theme = {
            'icon': '🚫',
            'secondary_icon': '🔍✗',
            'color': CONAN_COLORS['danger'],
            'bg_class': 'lie-detected',
            'verdict': 'كذب!',
            'verdict_en': 'MENSONGE',
            'message': 'هذا النص يبدو مضللاً',
            'conan_quote': '"الكذب له أقدام قصيرة" - كونان'
        }
    
    # Badge si ajusté par heuristiques
    conan_badge = ""
    if adjusted_by_heuristics:
        conan_badge = f"""
        <div style="
            background: {CONAN_COLORS['accent']};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        ">
            🕵️ تم التعديل بواسطة بصيرة كونان
        </div>
        """
    
    html = f"""
    <div class="{theme['bg_class']}">
        <!-- Icône principale -->
        <div class="verdict-icon">{theme['icon']}</div>
        
        <!-- Verdict -->
        <h2 class="verdict-text" style="color: {theme['color']};">
            {theme['verdict']}
        </h2>
        <div style="font-size: 1.1rem; color: {theme['color']}; opacity: 0.8; font-weight: 600;">
            {theme['message']}
        </div>
        
        {conan_badge}
        
        <!-- Badge de confiance -->
        <div class="confidence-badge">
            <div class="confidence-label">مستوى الثقة</div>
            <div class="confidence-value" style="color: {theme['color']};">
                {confidence:.1%}
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">
                {theme['secondary_icon']}
            </div>
        </div>
        
        <!-- Barre de progression -->
        <div class="magnifier-bar">
            <div class="magnifier-fill" style="
                width: {confidence*100}%;
                background: {theme['color']};
            "></div>
        </div>
        
        <!-- Statistiques -->
        <div class="stats-grid">
            <div class="stat-badge">
                <div class="stat-label">📝 الكلمات</div>
                <div class="stat-value">{stats.get('words', 0)}</div>
            </div>
            <div class="stat-badge">
                <div class="stat-label">🔤 الأحرف</div>
                <div class="stat-value">{stats.get('chars', 0)}</div>
            </div>
            <div class="stat-badge">
                <div class="stat-label">🤖 النموذج</div>
                <div class="stat-value" style="font-size: 0.9rem; text-transform: uppercase;">
                    {stats.get('model', 'N/A')}
                </div>
            </div>
        </div>
        
        <!-- Citation de Conan -->
        <div style="
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.6);
            border-radius: 0.8rem;
            border-right: 4px solid {theme['color']};
            font-style: italic;
            color: #334155;
            font-size: 0.95rem;
        ">
            {theme['conan_quote']}
        </div>
    </div>
    """
    
    return html


def generate_error_html(error_message: str, error_type: str = "warning") -> str:
    """
    Générer un message d'erreur style Detective Conan
    
    Args:
        error_message: Message d'erreur
        error_type: Type (warning, error, info)
        
    Returns:
        HTML formaté
    """
    
    themes = {
        'warning': {
            'icon': '⚠️',
            'color': CONAN_COLORS['warning'],
            'bg': 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
            'border': CONAN_COLORS['warning']
        },
        'error': {
            'icon': '❌',
            'color': CONAN_COLORS['danger'],
            'bg': 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
            'border': CONAN_COLORS['danger']
        },
        'info': {
            'icon': 'ℹ️',
            'color': CONAN_COLORS['primary'],
            'bg': 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
            'border': CONAN_COLORS['primary']
        }
    }
    
    theme = themes.get(error_type, themes['warning'])
    
    return f"""
    <div style="
        background: {theme['bg']};
        border: 3px solid {theme['border']};
        border-radius: 1.5rem;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{theme['icon']}</div>
        <div style="
            color: {theme['color']};
            font-size: 1.2rem;
            font-weight: 700;
        ">
            {error_message}
        </div>
        <div style="
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #64748b;
        ">
            🕵️ المحقق كونان يحتاج إلى مزيد من المعلومات
        </div>
    </div>
    """


def generate_loading_html() -> str:
    """Message de chargement style Conan"""
    return """
    <div style="
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 1.5rem;
        border: 3px solid #3b82f6;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem; animation: iconPulse 1.5s ease-in-out infinite;">
            🔍
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1e40af; margin-bottom: 0.5rem;">
            جاري التحقيق...
        </div>
        <div style="font-size: 1rem; color: #64748b;">
            المحقق كونان يفحص الأدلة
        </div>
    </div>
    """
