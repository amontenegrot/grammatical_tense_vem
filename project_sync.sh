#!/bin/bash

# --- VARIABLES ---
RUTA_WSL="$HOME/proyectos/grammatical_tense_vem/"

RUTA_COMPARTIDA="/mnt/c/Users/almontao/OneDrive/Escritorio/linwin_compartida/grammatical_tense_vem_mirror/"

echo "🚀 Preparando carpetas..."
# Este comando crea la carpeta en Windows automáticamente si no existe
mkdir -p "$RUTA_COMPARTIDA"

echo "🔄 Sincronizando scripts y base de datos a W11..."

rsync -av \
  --exclude='venv/' \
  --exclude='ds003020/' \
  "$RUTA_WSL" "$RUTA_COMPARTIDA"

# Verificamos si hubo algún error en rsync
if [ $? -eq 0 ]; then
    echo "✅ ¡Sincronización terminada con éxito!"
else
    echo "❌ ERROR: Revisa el mensaje de arriba. ¿Escribiste bien la ruta de Windows?"
fi