# Modelamiento Voxel-wise del Tiempo Gramatical Finito en fMRI Naturalista

Este repositorio contiene la tubería computacional (pipeline) y el marco metodológico para evaluar la varianza predictiva única del tiempo gramatical finito (Pasado vs. No Pasado) durante la comprensión del habla naturalista. 

El proyecto implementa Modelos de Codificación Voxel-wise (Voxelwise Encoding Models - VEM) analizando el conjunto de datos `ds003020` de OpenNeuro. La arquitectura extrae representaciones lingüísticas de múltiples niveles (fonológico, léxico-estadístico, léxico-categorial, sintáctico y semántico latente), aplica modelamiento hemodinámico y estima mapas corticales de alta fidelidad, previniendo la fuga de datos (data leakage) y aplicando un control estadístico riguroso.

## Decisiones Técnicas y de Optimización
El pipeline ha sido refactorizado para garantizar un rendimiento óptimo bajo estándares de Ingeniería de Software (PEP 8) y Open Science:
*   **Centralización Paramétrica:** Todos los hiperparámetros (ej. umbrales estadísticos, retardos HRF, semillas de reproducibilidad, filtros NLP) se administran desde una única fuente de la verdad (`src/config.py`).
*   **Gestión Extrema de Memoria (Anti-Thrashing):** Las matrices predictoras y de fMRI superan con facilidad los 30 GB en RAM. Se implementó coerción temprana a `float32`, recolección explícita de basura (`gc.collect()`) y estandarización matemática *in-place* para evitar el colapso del sistema operativo (OOM Killer).
*   **Resolución en Espacio Primal vs. Dual:** Ante la explosión combinatoria de calcular matrices de Kernel masivas (27,000 muestras x 27,000 muestras) en el enfoque *Banded Ridge*, el pipeline principal migró hacia una resolución en el Espacio Primal (*Standard Ridge*), reduciendo los tiempos de cómputo de horas a escasos minutos por participante.

## Requisitos del Sistema

*   **Sistema Operativo**: Linux / WSL2 (Se recomienda Ubuntu).
*   **Hardware**: Procesador (CPU) multinúcleo con alta capacidad de subprocesos (ej. Intel Core i9, AMD Ryzen 9). Se requiere un mínimo de 32 GB de RAM (Se recomiendan estrictamente 64 GB).
*   **Aceleración de Hardware**: El entrenamiento principal utiliza librerías C de bajo nivel (OpenBLAS/MKL) que paralelizan las operaciones matriciales a través del 100% de los núcleos físicos de la CPU. No se recomienda el uso de GPU para la regresión Ridge masiva debido a la rápida saturación de la VRAM frente a tensores fMRI completos.
*   **Versión de Python**: 3.12 o superior.

## Configuración e Instalación

1. Clone el repositorio y navegue hasta la raíz del proyecto.
2. Cree un entorno virtual y actívelo:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instale las dependencias requeridas (asegúrese de incluir `himalaya`, `spacy`, `scipy`, `pandas`, `pyarrow`, `pycortex`, `wordfreq` y `scikit-learn`).
4. Descargue el modelo Transformer de spaCy necesario para la extracción sintáctica y categorial:
   ```bash
   python -m spacy download en_core_web_trf
   ```

## Estructura de Directorios (Orientada a BIDS y Open Science)

```text
grammatical_tense_vem/
├── artifacts/                  # Modelos serializados de Machine Learning (ej. TF-IDF + SVD)
├── data/
│   ├── ds003020/               # Dataset original BIDS (fMRI hf5, TextGrids, estímulos, pycortex-db)
│   └── processed/              # Derivados del proyecto: matrices temporales, mapas web y resultados
├── db/                         # Bases de datos SQLite de auditoría/presentación
│   └── csv_reports/            # Exportaciones tabulares legibles de las auditorías
├── scripts/                    # Orquestadores de ejecución modular
└── src/                        # Código fuente de la arquitectura (Estándar PEP 8)
    ├── features/               # NLP, acústica y transformaciones neurovasculares
    ├── models/                 # Lógica predictiva (Primal Ridge y Dual Banded Ridge)
    ├── config.py               # Centralizador global de hiperparámetros y rutas
    ├── data_loader.py          # Gestor de memoria optimizado para HDF5
    └── db_manager.py           # Conector seguro para bases de datos relacionales
```

## Guía de Ejecución del Pipeline

**NOTA CRÍTICA**: Todos los scripts deben ejecutarse desde la raíz absoluta del proyecto (`grammatical_tense_vem/`). Debe anteponer la variable de entorno `PYTHONPATH=.` para asegurar que el intérprete de Python mapee correctamente los módulos internos.

El flujo de trabajo es completamente modular e incluye puntos de control (checkpointing). Si un script se interrumpe, volver a ejecutar el comando omitirá los datos ya transformados y reanudará la ejecución de forma segura.

### Fase 1: Auditoría de Datos y Entrenamiento Global

**1. Auditoría del Conjunto de Datos**
Cruza los TextGrids disponibles, los estímulos acústicos y las series fMRI, aplicando criterios de exclusión. Almacena el mapeo analítico en SQLite y exporta reportes a `db/csv_reports/`.
```bash
PYTHONPATH=. python scripts/01a_audit_dataset.py
```

**2. Entrenamiento Semántico LSA**
Extrae las cláusulas de todo el corpus y entrena un modelo global de Análisis Semántico Latente. Los parámetros de filtrado (stop-words de dominio, patrones de ruido acústico, dimensiones LSA) se importan dinámicamente desde la configuración central.
```bash
PYTHONPATH=. python scripts/01b_train_semantic_model.py
```

**3. Auditoría de la Historia de Prueba**
Valida la viabilidad metodológica del Hold-out Test Set (ej. `wheretheressmoke`) garantizando que todos los participantes la hayan escuchado, y detecta exposiciones repetidas para el cálculo del "Techo de Ruido" (Noise Ceiling).
```bash
PYTHONPATH=. python scripts/01c_audit_test_story.py
```

### Fase 2: Extracción de Características (Representación del Estímulo)

**4. Extracción de Espacios a Alta Resolución**
Proyecta las propiedades lingüísticas en matrices continuas de alta resolución (100 Hz). Implementa una exportación dual: matrices `.parquet` ultraligeras para el pipeline de ML, y una exportación truncada a `.sqlite` de una historia de muestra para presentaciones interactivas ante jurados.
```bash
PYTHONPATH=. python scripts/02_extract_features.py
```

### Fase 3: Modelamiento Temporal Neurovascular

**5. Convolución Hemodinámica y Reducción de Resolución**
Aplica una Función de Respuesta Hemodinámica (HRF) Double-Gamma mediante Transformada Rápida de Fourier (FFT), o alternativamente, una Respuesta al Impulso Finito (FIR). Posteriormente, ejecuta un filtro polifásico anti-aliasing para reducir la resolución temporal y alinear los regresores al TR del escáner (2.0s). Utiliza detección dinámica de núcleos para multiprocesamiento.
```bash
PYTHONPATH=. python scripts/03_apply_hrf.py
```

### Fase 4: Modelamiento Predictivo (Machine Learning)

El proyecto ofrece dos orquestadores matemáticos, ambos con validación cruzada y validación estadística por permutación de desplazamiento circular.

**6A. Regresión de Cresta Estándar (Recomendado - Enfoque Primal)**
Resuelve el sistema en el espacio de características (45 dimensiones). Implementa procesamiento espacial por lotes (Voxel Chunking) procesando bloques de 10,000 vóxeles a la vez. Esto evita crear tensores tridimensionales de error que colapsan la RAM, manteniendo el uso de memoria inferior a 10 GB y disparando la eficiencia del procesador.
```bash
PYTHONPATH=. python scripts/04b_train_voxelwise_standard_ridge.py
```

**6B. Regresión de Cresta en Bandas (Alternativa - Enfoque Dual)**
Calcula Kernels independientes por cada espacio lingüístico para penalizaciones Múltiples. Debido a la cantidad de muestras temporales (+27,000 en algunos sujetos), su costo computacional es masivo.
```bash
PYTHONPATH=. python scripts/04a_train_voxelwise_banded_ridge.py
```

### Fase 5: Visualización Cortical Interactiva

**7. Generación de Mapas Corticales Enmascarados**
Lee los resultados de la varianza predictiva única ($\Delta R^2$) y aplica un enmascaramiento estadístico estricto (Máscara FDR). Los vóxeles con un valor p corregido superior a `STATISTICAL_ALPHA` (ej. 0.05) se renderizan transparentes para eliminar el ruido visual. Exporta un visualizador multicapa (Global, Restringido y Delta R2) estático (HTML/WebGL) a `data/processed/cortical_maps_web/`.
```bash
PYTHONPATH=. python scripts/05_generate_cortical_maps.py
```
*Nota: Si se le pasa una cadena de texto (ej. "sub-UTS03") en el código fuente, iniciará un servidor en vivo en el navegador. Si se le pasa una lista, procesará el lote completo en modo silencioso.*
```