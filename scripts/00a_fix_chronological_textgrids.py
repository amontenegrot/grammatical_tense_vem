# scripts/00a_fix_chronological_textgrids.py
"""Convierte TextGrids cronológicos de Praat a formato estándar ooTextFile usando solo tgt."""
import re
import shutil
import sys
from pathlib import Path

# Garantizar acceso al proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tgt
from src.config import DIR_TEXTGRIDS

STORIES_TO_FIX = ['legacy', 'exorcism']


def parse_chronological_praat(filepath: Path) -> tgt.TextGrid:
    """Parsea un TextGrid en formato cronológico y lo convierte en objeto tgt.TextGrid."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or "chronological" not in lines[0].lower():
        raise ValueError("El archivo no tiene cabecera de TextGrid cronológico.")

    # 1. Metadatos globales (Línea 2 y 3)
    # Ej: "0.0124716553288 955.021095485 ! Time domain."
    time_domain_match = re.match(r'^([\d\.\-eE]+)\s+([\d\.\-eE]+)', lines[1])
    global_xmin = float(time_domain_match.group(1))
    global_xmax = float(time_domain_match.group(2))

    n_tiers = int(lines[2].split()[0])

    # 2. Definición de Tiers (Líneas 3 a 3 + n_tiers)
    # Ej: '"IntervalTier" "words" 0.0124716553288 955.021095485'
    tiers_info = {}
    current_line = 3
    for tier_idx in range(1, n_tiers + 1):
        line = lines[current_line]
        tier_match = re.match(r'^"([^"]+)"\s+"([^"]+)"\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', line)
        if tier_match:
            tier_type = tier_match.group(1)
            tier_name = tier_match.group(2)
            tier_xmin = float(tier_match.group(3))
            tier_xmax = float(tier_match.group(4))
            tiers_info[tier_idx] = {
                'name': tier_name,
                'type': tier_type,
                'xmin': tier_xmin,
                'xmax': tier_xmax,
                'intervals': []
            }
        current_line += 1

    # 3. Intervalos cronológicos (Líneas restantes)
    # Ej: '1 0.0124716553288 0.5 "palabra"'
    interval_pattern = re.compile(r'^(\d+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+"(.*)"$')

    for line in lines[current_line:]:
        match = interval_pattern.match(line)
        if match:
            t_idx = int(match.group(1))
            start_t = float(match.group(2))
            end_t = float(match.group(3))
            text = match.group(4).replace('""', '"')  # Desescapar comillas dobles

            if t_idx in tiers_info:
                tiers_info[t_idx]['intervals'].append((start_t, end_t, text))

    # 4. Construir objeto TextGrid de tgt
    tg = tgt.TextGrid()
    for t_idx in sorted(tiers_info.keys()):
        t_data = tiers_info[t_idx]
        tier = tgt.IntervalTier(
            start_time=t_data['xmin'],
            end_time=t_data['xmax'],
            name=t_data['name']
        )
        for start_t, end_t, text in t_data['intervals']:
            tier.add_interval(tgt.Interval(start_time=start_t, end_time=end_t, text=text))
        tg.add_tier(tier)

    return tg


def fix_all():
    print("=== REPARACIÓN DE TEXTGRIDS CRONOLÓGICOS (SIN NUEVAS LIBRERÍAS) ===\n")
    for story in STORIES_TO_FIX:
        tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
        if not tg_path.exists():
            print(f"[OMITIDO] No encontrado: {tg_path}")
            continue

        with open(tg_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()

        if "chronological" in first_line.lower():
            print(f"Procesando: {story}.TextGrid (Formato cronológico detectado)...")
            
            # 1. Crear respaldo .bak
            bak_path = tg_path.with_suffix(".TextGrid.bak")
            if not bak_path.exists():
                shutil.copyfile(tg_path, bak_path)
                print(f" -> Respaldo de seguridad creado: {bak_path.name}")

            # 2. Parsear y escribir en formato estándar ooTextFile
            try:
                tg = parse_chronological_praat(tg_path)
                tgt.io.write_to_file(tg, str(tg_path), format='long')
                print(f" -> [ÉXITO] Convertido a estándar Praat: {story}.TextGrid")
            except Exception as e:
                print(f" -> [ERROR] Falló la conversión: {e}")
        else:
            print(f"{story}.TextGrid ya está en formato estándar.")


if __name__ == "__main__":
    fix_all()
