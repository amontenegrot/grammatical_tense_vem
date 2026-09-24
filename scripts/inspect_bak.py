# scripts/inspect_bak.py
import sys
from pathlib import Path

# --- GARANTIZAR QUE PYTHON ENCUENTRE EL MÓDULO 'src' ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------------------------------------


from src.config import DIR_TEXTGRIDS

bak_path = DIR_TEXTGRIDS / "legacy.TextGrid.bak"
if not bak_path.exists():
    bak_path = DIR_TEXTGRIDS / "legacy.TextGrid"

print(f"Inspeccionando: {bak_path}\n")
with open(bak_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i in range(25):
        line = f.readline()
        if not line:
            break
        print(f"Línea {i+1:02d}: {repr(line)}")
