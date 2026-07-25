# Voxelwise Encoding Modeling of Finite Grammatical Tense in Naturalistic fMRI

This repository contains the computational pipeline and methodological framework to evaluate the unique predictive variance of finite grammatical tense (Past vs. Non-Past) during naturalistic speech comprehension. The project implements a Voxelwise Encoding Model (VEM) using Banded Ridge Regression, analyzing the OpenNeuro dataset `ds003020`.

The pipeline extracts multi-level linguistic representations (phonological, lexical-statistical, lexical-categorical, syntactic, and latent semantic), applies hemodynamic modeling, and estimates cortical mapping while preventing data leakage and controlling for multi-collinearity.

## System Requirements

*   **Operating System**: Linux / WSL2 (Ubuntu recommended).
*   **Hardware**: Multi-core CPU (e.g., Intel Core i9 or similar) with high multi-threading capacity. Minimum 32 GB RAM (64 GB highly recommended). GPU is not required nor recommended due to VRAM limitations and memory thrashing during high-dimensional Ridge Regression.
*   **Python Version**: 3.12 or higher.

## Setup and Installation

1. Clone the repository and navigate to the project root.
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies (ensure `himalaya`, `spacy`, `scipy`, `pandas`, `pyarrow`, `pycortex`, and `scikit-learn` are included).
4. Download the necessary spaCy Transformer model:
   ```bash
   python -m spacy download en_core_web_trf
   ```

## Directory Structure

```text
grammatical_tense_vem/
├── data/
│   ├── ds003020/               # Original dataset (fMRI, TextGrids, stimuli)
│   └── derivatives/            # Generated features and voxelwise results
├── db/                         # SQLite database and analytical CSV reports
├── docs/                       # Static HTML PyCortex exports
├── models/                     # Serialized Machine Learning pipelines (e.g., LSA)
├── scripts/                    # Execution orchestrators
└── src/                        # Core architecture modules
    ├── features/               # Feature extractors and hemodynamic processing
    └── models/                 # Ridge regression and statistical validation
```

## Pipeline Execution Guide

**CRITICAL NOTE**: All scripts must be executed from the absolute root of the project (`grammatical_tense_vem/`). You must prepend the environment variable `PYTHONPATH=.` to ensure the Python interpreter correctly maps the `src/` modules.

The pipeline is fully modular, sequential, and includes atomic checkpointing. If a script is interrupted (e.g., power loss), re-running the command will automatically skip previously processed data and resume execution.

### Phase 1: Data Auditing and Global Training

**1. Dataset Audit**
Cross-references available TextGrids, acoustic stimuli, and fMRI files (`.hf5`), storing the analytical mapping in a local SQLite database.
```bash
PYTHONPATH=. python scripts/01_audit_dataset.py
```

**2. Semantic LSA Training**
Trains a global Latent Semantic Analysis pipeline (TF-IDF + TruncatedSVD) across the entire dataset to capture dynamic semantic states, filtering conversational stop-words.
```bash
PYTHONPATH=. python scripts/01b_train_semantic_model.py
```

**3. Test Story Audit**
Validates the viability of the designated test set (e.g., `wheretheressmoke`) by querying the database for play counts per subject to ensure accurate cross-validation availability.
```bash
PYTHONPATH=. python scripts/01c_audit_test_story.py
```

### Phase 2: Feature Extraction (Stimulus Representation)

**4. Feature Extraction**
Parses transcripts and acoustic properties into high-resolution continuous time-series (100 Hz). Implements a Single-Pass Inference architecture for the Transformer NLP model to prevent PyTorch deadlocks and minimize RAM overhead.
```bash
PYTHONPATH=. python scripts/02_extract_features.py
```

### Phase 3: Temporal Modeling

**5. Hemodynamic Convolution**
Applies a Double-Gamma Hemodynamic Response Function (HRF) using Fast Fourier Transform (FFT) convolution, and downsampels the resolution to match the fMRI Repetition Time (TR = 2.0s) using an anti-aliasing polyphase filter. This step is executed in parallel across all logical CPU cores.
```bash
PYTHONPATH=. python scripts/03_apply_hrf.py
```

### Phase 4: Predictive Modeling (Machine Learning)

**6. Voxelwise Regression**
Executes Banded Ridge Regression per subject. Computes the global model, the restricted model (ablation of the finite tense space), the unique predictive variance ($\Delta R^2$), and conducts statistical validation via circular shift permutations and False Discovery Rate (FDR) correction.
```bash
PYTHONPATH=. python scripts/04_train_voxelwise_models.py
```
*Note: This is computationally intensive. Ensure parameters (`n_iter`, `alphas`, and `PERMUTATIONS=1000`) are properly set in the source code prior to full execution.*

### Phase 5: Cortical Visualization

**7. Generate Interactive Cortical Maps**
Projects the predictive variance results onto the subject's native 3D anatomical surface using `pycortex`. Automatically hosts a local dynamic web server for interactive exploration of the brain model.
```bash
PYTHONPATH=. python scripts/05_generate_cortical_maps.py
```