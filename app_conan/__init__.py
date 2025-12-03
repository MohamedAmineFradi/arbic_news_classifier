"""Module app - Detective Conan Fake News Detector"""

from .conan_interface import (
    create_conan_interface,
    launch_app,
    ConanFakeNewsDetector
)

from .model_handler import ModelHandler
from .result_generator import (
    generate_result_html,
    generate_error_html,
    generate_loading_html
)

from .theme_conan import (
    CSS_CONAN_THEME,
    CONAN_COLORS,
    CONAN_EXAMPLES,
    get_header_html,
    get_footer_html
)

__all__ = [
    'create_conan_interface',
    'launch_app',
    'ConanFakeNewsDetector',
    'ModelHandler',
    'generate_result_html',
    'CSS_CONAN_THEME',
    'CONAN_COLORS',
    'CONAN_EXAMPLES'
]

__version__ = '2.0.0'
__author__ = 'Detective Conan Team'
