"""
Point d'entrée principal pour l'application Detective Conan
Utilise la nouvelle architecture modulaire
"""

import sys
from pathlib import Path

# Assurer que le dossier racine du projet est sur sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_conan.conan_interface import launch_app

if __name__ == "__main__":
    launch_app(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
