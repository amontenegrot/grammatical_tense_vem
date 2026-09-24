# scripts/00a_fix_chronological_textgrids.py
"""Convierte TextGrids cronológicos de Praat a formato estándar ooTextFile en 2 líneas por intervalo."""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tgt
from src.config import DIR_TEXTGRIDS

STORIES_TO_FIX = ['legacy', 'exorcism']


def parse_chronological_praat(filepath: Path) -> tgt.TextGrid:
    """Parsea el formato cronológico de Praat (pares de líneas: tiempos y texto)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or "chronological" not in lines[0].lower():
        raise ValueError("No contiene cabecera cronológica.")

    # 1. Metadatos globales (Líneas 2 y 3)
    time_domain_match = re.match(r'^([\d\.\-eE]+)\s+([\d\.\-eE]+)', lines[1])
    global_xmin = float(time_domain_match.group(1))
    global_xmax = float(time_domain_match.group(2))
    n_tiers = int(lines[2].split()[0])

    # 2. Definición de Tiers
    tiers_info = {}
    idx = 3
    for tier_idx in range(1, n_tiers + 1):
        line = lines[idx]
        tier_match = re.match(r'^"([^"]+)"\s+"([^"]+)"\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', line)
        if tier_match:
            tiers_info[tier_idx] = {
                'type': tier_match.group(1),
                'name': tier_match.group(2),
                'xmin': float(tier_match.group(3)),
                'xmax': float(tier_match.group(4)),
                'intervals': []
            }
        idx += 1

    # 3. Intervalos en pares de líneas (Línea A: tiempos, Línea B: texto)
    timing_pattern = re.compile(r'^(\d+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)$')

    while idx < len(lines):
        line_a = lines[idx]
        idx += 1
        
        match = timing_pattern.match(line_a)
        if not match:
            continue

        t_idx = int(match.group(1))
        start_t = float(match.group(2))
        end_t = float(match.group(3))

        if idx >= len(lines):
            break

        line_b = lines[idx]
        idx += 1

        # Limpiar comillas del texto
        if line_b.startswith('"') and line_b.endswith('"'):
            text = line_b[1:-1].replace('""', '"')
        else:
            text = line_b.replace('"', '')

        if t_idx in tiers_info:
            tiers_info[t_idx]['intervals'].append((start_t, end_t, text))

    # 4. Construir el TextGrid estándar de tgt
    tg = tgt.TextGrid()
    for t_idx in sorted(tiers_info.keys()):
        t_data = tiers_info[t_idx]
        # Ordenar cronológicamente dentro de cada tier
        t_data['intervals'].sort(key=lambda x: x[0])
        
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
    print("=== REPARACIÓN DE TEXTGRIDS CRONOLÓGICOS (ESTRUCTURA DE 2 LÍNEAS) ===\n")
    for story in STORIES_TO_FIX:
        # Siempre leer desde el respaldo original .bak si existe
        bak_path = DIR_TEXTGRIDS / f"{story}.TextGrid.bak"
        tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
        
        source_path = bak_path if bak_path.exists() else tg_path
        if not source_path.exists():
            print(f"[OMITIDO] No encontrado: {source_path}")
            continue

        print(f"Procesando '{story}' desde '{source_path.name}'...")
        try:
            tg = parse_chronological_praat(source_path)
            # Guardar en formato estándar ooTextFile de Praat
            tgt.io.write_to_file(tg, str(tg_path), format='long')
            
            # Estadísticas de verificación rápida
            for tier in tg.tiers:
                n_words = sum(1 for iv in tier.intervals if iv.text.strip() != "")
                print(f" -> Tier '{tier.name}': {len(tier.intervals)} intervalos ({n_words} con texto)")
            print(f" -> [ÉXITO] Archivo {tg_path.name} reconstruido.\n")
        except Exception as e:
            print(f" -> [ERROR]: {e}\n")


if __name__ == "__main__":
    fix_all()
