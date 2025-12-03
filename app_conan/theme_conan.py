"""Thème Detective Conan pour l'interface Gradio"""

# Couleurs du thème Detective Conan
CONAN_COLORS = {
    'primary': '#1e40af',      # Bleu foncé (costume de Conan)
    'secondary': '#dc2626',    # Rouge (nœud papillon)
    'accent': '#f59e0b',       # Orange (badge détective)
    'success': '#059669',      # Vert (vérité découverte)
    'warning': '#ea580c',      # Orange foncé
    'danger': '#b91c1c',       # Rouge foncé (mensonge détecté)
    'light': '#f0f9ff',        # Bleu très clair
    'dark': '#1e293b',         # Gris foncé
    'magnifier': '#fbbf24',    # Doré (loupe)
}

CSS_CONAN_THEME = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Press+Start+2P&display=swap');
    
    .gradio-container {{
        font-family: 'Cairo', 'Segoe UI', sans-serif !important;
        direction: rtl;
        background: linear-gradient(180deg, {CONAN_COLORS['light']} 0%, #e0f2fe 50%, #dbeafe 100%);
        min-height: 100vh;
    }}
    
    /* En-tête style Detective Conan */
    .conan-header {{
        background: linear-gradient(135deg, {CONAN_COLORS['primary']} 0%, #3b82f6 50%, {CONAN_COLORS['secondary']} 100%);
        padding: 2.5rem;
        border-radius: 1.5rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(30, 64, 175, 0.4);
        position: relative;
        overflow: hidden;
        border: 3px solid {CONAN_COLORS['magnifier']};
    }}
    
    .conan-header::before {{
        content: '🔍';
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 120px;
        opacity: 0.1;
        transform: rotate(-15deg);
    }}
    
    .conan-header::after {{
        content: '🕵️';
        position: absolute;
        bottom: -20px;
        left: -20px;
        font-size: 100px;
        opacity: 0.1;
        transform: rotate(15deg);
    }}
    
    .conan-title {{
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        letter-spacing: 1px;
    }}
    
    .conan-subtitle {{
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.8rem;
        font-weight: 600;
    }}
    
    .conan-motto {{
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 0.5rem;
        font-style: italic;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 0.8rem;
        margin-top: 1rem;
    }}
    
    /* Cartes style détective */
    .detective-card {{
        background: white;
        border-right: 5px solid {CONAN_COLORS['primary']};
        padding: 1.8rem;
        border-radius: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        position: relative;
        height: 100%;
    }}
    
    .detective-card::before {{
        content: '🔎';
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 2rem;
        opacity: 0.15;
    }}
    
    .detective-card h3 {{
        color: {CONAN_COLORS['primary']};
        margin-top: 0;
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* Résultats style révélation */
    .truth-revealed {{
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 3px solid {CONAN_COLORS['success']};
        border-radius: 1.5rem;
        padding: 2.5rem;
        text-align: center;
        position: relative;
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.3);
        animation: revealTruth 0.6s ease-out;
    }}
    
    .lie-detected {{
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%);
        border: 3px solid {CONAN_COLORS['danger']};
        border-radius: 1.5rem;
        padding: 2.5rem;
        text-align: center;
        position: relative;
        box-shadow: 0 10px 30px rgba(185, 28, 28, 0.3);
        animation: detectLie 0.6s ease-out;
    }}
    
    @keyframes revealTruth {{
        from {{
            opacity: 0;
            transform: scale(0.9) translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: scale(1) translateY(0);
        }}
    }}
    
    @keyframes detectLie {{
        0% {{
            opacity: 0;
            transform: scale(0.9);
        }}
        50% {{
            transform: scale(1.02);
        }}
        100% {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .verdict-icon {{
        font-size: 5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 6px 12px rgba(0,0,0,0.2));
        animation: iconPulse 2s ease-in-out infinite;
    }}
    
    @keyframes iconPulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
    }}
    
    .verdict-text {{
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    
    .confidence-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.9);
        padding: 1rem 2rem;
        border-radius: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .confidence-label {{
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }}
    
    .confidence-value {{
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
    }}
    
    /* Barre de progression style loupe */
    .magnifier-bar {{
        background: rgba(255,255,255,0.7);
        height: 12px;
        border-radius: 6px;
        overflow: hidden;
        margin: 1.5rem auto;
        width: 85%;
        border: 2px solid rgba(0,0,0,0.1);
        position: relative;
    }}
    
    .magnifier-bar::after {{
        content: '🔍';
        position: absolute;
        top: -8px;
        right: -5px;
        font-size: 1.5rem;
        animation: magnifierMove 3s ease-in-out infinite;
    }}
    
    @keyframes magnifierMove {{
        0%, 100% {{ right: -5px; }}
        50% {{ right: 10px; }}
    }}
    
    .magnifier-fill {{
        height: 100%;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }}
    
    /* Statistiques style badge détective */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }}
    
    .stat-badge {{
        background: rgba(255,255,255,0.85);
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        border: 2px solid {CONAN_COLORS['accent']};
        transition: transform 0.2s ease;
    }}
    
    .stat-badge:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }}
    
    .stat-label {{
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }}
    
    .stat-value {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {CONAN_COLORS['primary']};
    }}
    
    /* Bouton analyse style Conan */
    .analyze-btn {{
        background: linear-gradient(135deg, {CONAN_COLORS['primary']} 0%, #2563eb 100%) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 1rem 2.5rem !important;
        border-radius: 0.8rem !important;
        border: 3px solid {CONAN_COLORS['magnifier']} !important;
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    
    .analyze-btn:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.5) !important;
    }}
    
    /* Zone de texte */
    #input_box {{
        border: 3px solid {CONAN_COLORS['primary']} !important;
        border-radius: 1rem !important;
        font-size: 1.1rem !important;
        background: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }}
    
    #input_box:focus {{
        border-color: {CONAN_COLORS['accent']} !important;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3) !important;
    }}
    
    /* Footer style manga */
    .conan-footer {{
        text-align: center;
        padding: 2rem;
        background: linear-gradient(to right, rgba(30, 64, 175, 0.05), rgba(220, 38, 38, 0.05));
        border-radius: 1rem;
        margin-top: 2rem;
        border: 2px dashed {CONAN_COLORS['primary']};
    }}
    
    .warning-box {{
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid {CONAN_COLORS['warning']};
        padding: 1.5rem;
        border-radius: 1rem;
        color: #92400e;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.2);
    }}
    
    /* Exemples style cas d'enquête */
    .examples-container {{
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid {CONAN_COLORS['primary']};
        margin-top: 2rem;
    }}
    
    .case-example {{
        background: linear-gradient(to right, #f0f9ff, #e0f2fe);
        padding: 1rem;
        border-radius: 0.5rem;
        border-right: 4px solid {CONAN_COLORS['accent']};
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .case-example:hover {{
        background: linear-gradient(to right, #e0f2fe, #bae6fd);
        transform: translateX(-5px);
    }}
</style>
"""

def get_header_html():
    """Retourne le HTML de l'en-tête Detective Conan"""
    return f"""
    <div class="conan-header">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🕵️‍♂️🔍</div>
        <h1 class="conan-title">المحقق كونان للأخبار</h1>
        <p class="conan-subtitle">محقق الحقيقة في عالم الأخبار المزيفة</p>
        <p class="conan-motto">"الحقيقة دائماً واحدة!" - المحقق كونان</p>
    </div>
    """

def get_input_card_html():
    """Carte d'input style dossier d'enquête"""
    return """
    <div class="detective-card">
        <h3>📋 ملف القضية - النص المراد التحقيق منه</h3>
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            قم بإدخال النص أو الخبر الذي تريد التحقق من مصداقيته.<br>
            <strong>المحقق كونان سيكشف الحقيقة!</strong>
        </p>
    </div>
    """

def get_result_card_html():
    """Carte des résultats style révélation"""
    return """
    <div class="detective-card">
        <h3>🎯 نتيجة التحقيق</h3>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
            تقرير المحقق كونان حول مصداقية النص
        </p>
    </div>
    """

def get_footer_html():
    """Footer avec avertissement style Conan"""
    return """
    <div class="conan-footer">
        <div class="warning-box">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️ 🔍</div>
            <strong>تنبيه المحقق كونان:</strong><br>
            هذه الأداة مساعدة للتحقيق وتستخدم خوارزميات الذكاء الاصطناعي.<br>
            مثل المحقق كونان، يجب دائماً جمع الأدلة من مصادر متعددة للوصول إلى الحقيقة!
        </div>
        <div style="margin-top: 1.5rem; color: #64748b; font-size: 0.9rem;">
            <strong>🎭 مستوحى من Detective Conan</strong> - "لا توجد جريمة كاملة!"
        </div>
    </div>
    """

# Exemples style cas d'enquête Conan
CONAN_EXAMPLES = [
    ["📰 القضية #001: أعلنت وزارة الصحة رسمياً عن انخفاض معدلات الإصابة بنسبة 20% هذا الأسبوع وفقاً للتقرير الأسبوعي المنشور على الموقع الرسمي."],
    ["🚨 القضية #002: عاجل وخطير! كائنات فضائية تهبط في الأهرامات والحكومة تخفي الأمر عن الشعب! شاهد الصور الحصرية!"],
    ["💊 القضية #003: اكتشاف عشبة سحرية تشفي جميع الأمراض في 24 ساعة! الأطباء يخفون هذا السر منذ سنوات!"],
    ["⚽ القضية #004: فاز المنتخب الوطني في مباراة الأمس بنتيجة 3-1 في إطار تصفيات كأس العالم حسب تقرير الاتحاد الدولي لكرة القدم."],
    ["👽 القضية #005: تسريب خطير: ديناصورات عملاقة تعيش تحت الأرض وسيخرجون قريباً! استعدوا لنهاية العالم!"],
]
