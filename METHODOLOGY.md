# FULL REPORT ON METHODOLOGICAL DECISIONS AND COMPUTATIONAL ARCHITECTURE

**Project:** Voxel-wise Modeling of Finite Grammatical Tense in Naturalistic fMRI.  
**Dataset:** ds003020 (Preprocessed fMRI HDF5 derivatives and TextGrid aligned transcripts).

This document comprehensively details each stage of the workflow, the evaluated options, the theoretical justification underlying the linguistic extraction, and the software engineering decisions implemented to optimize computational performance.

---

## 1. Study Design and General Workflow Architecture

The study adopts an observational design based on open-access secondary data, investigating continuous speech comprehension in naturalistic conditions without the need to collect new neurofunctional data. The project's computational architecture was divided into independent modules designed to maintain a strict separation between **Feature Extraction (Stimulus Representation)**, **Temporal Modeling**, **Predictive Estimation (Machine Learning)**, and **Visualization**.

### 1.1 Infrastructure and Resilience Decisions
* **Storage Formats (I/O):** To handle the high read/write load of dense matrices, CSV files were replaced with the `.parquet` binary format using the `pyarrow` engine, which preserves native data types (e.g., `int8`, `float32`) and drastically reduces storage usage and disk bottlenecks.
* **Atomic Checkpointing System:** Due to high computational demands, all orchestrators implement a file existence validation system. If an interruption occurs (e.g., power loss), the system identifies the last processed milestone and resumes computation exclusively on missing data, safeguarding days of continuous processing.

---

## 2. Empirical Corpus Auditing and Test Configuration

Before stimulus processing, an automated auditing engine backed by a relational database (SQLite) was implemented.
* **$O(1)$ Acoustic Extraction:** Instead of decoding heavy audio matrices into memory, the duration of each acoustic stimulus was determined by reading the metadata directly from the `.wav` headers, allowing the entire corpus duration to be extracted in fractions of a second.
* **Test Story Validation:** The story *"wheretheressmoke"* was designated as the independent test set to evaluate model reliability and approximate the Noise Ceiling. A database query statistically validated its multiple playbacks per participant, identifying listening asymmetries that will guide the final cross-validation.

---

## 3. Control Spaces Construction (Features)

To rigorously separate linguistic representation, matrices were initially built at a high temporal resolution (100 Hz), preserving acoustic fidelity prior to hemodynamic transformation.

### 3.1. NLP Inference Architecture (Single-Pass Inference)
* **The Problem:** Categorical, syntactic, semantic, and grammatical tense extraction requires processing text using a massive language model (RoBERTa Transformer, `en_core_web_trf` from spaCy). Deploying this model in parallel using multiple cores would cause a RAM collapse (over 20 GB in model weights) and deadlocks due to PyTorch's internal thread management.
* **The Solution:** A **Single-Pass Inference** was implemented. Text is processed exactly once by the Transformer, and the resulting object (`Doc`) is sequentially injected into all extractor classes. This accelerated the NLP process by 75% while keeping RAM usage safe and efficient.

### 3.2. Phonological Space
* **Action:** Conversion of ARPABET transcripts into a 14-dimensional binary temporal matrix (`np.int8`).
* **Methodological Decision:** A hybrid representation of articulatory and distinctive features (e.g., stop, fricative, voiced, bilabial) was chosen over a *one-hot encoding* of isolated phonemes.
* **Justification:** It perfectly controls acoustic-motor variance in the primary and superior auditory cortex. It isolates the phonological load from the grammatical suffixes of interest (e.g., the fricativity/voicing of /-s/ and /-z/ in present tense, or the stops /-t/ and /-d/ in past tense), preventing the model from confounding acoustic processing with grammatical inflection.

### 3.3. Lexical-Statistical Space
* **Action:** Extraction of Occurrence (1/0), Log-Frequency (Zipf), Word Length in characters, and Duration in seconds.
* **Methodological Decision:** The frequency metric was calculated strictly on the **surface form** of the spoken word and not on its lemma.
* **Justification:** Its goal is to control the bottom-up neurocognitive effort of decoding and lexical access. Finite forms have disparate distributions (e.g., "is" is significantly more frequent than "was"). Calculating frequency on the surface form absorbs this statistical bias, preventing the regression from falsely attributing the variance of lexical access ease to grammatical tense.

### 3.4. Lexical-Categorical Space (POS Tagging)
* **Action:** Grammatical classification via NLP into 8 categorical dimensions.
* **Methodological Decision 1 (Acoustic Punctuation):** Syntax was simulated based on empirical thresholds derived from pause and silence durations in the corpus. Pauses between 0.35s and 0.65s were converted into commas; pauses >0.65s (75th percentile) into periods. High-variance metalinguistic tags were ignored.
* **Justification 1:** Provides the neural network with a context featuring biologically informed clausal boundaries, avoiding over-segmentation caused by articulatory hesitations and optimizing syntactic dependency inference.
* **Methodological Decision 2 (Exclusion of Finite Verbs):** The non-finite verbs category (`non_finite_verb`: infinitives, participles, gerunds) was included, but all finite verbs were deliberately excluded.
* **Justification 2:** Prevents collinearity. Controls the processing of general verbal material without absorbing the variance of the finite inflectional contrast, which is the exclusive phenomenon of interest in this study.

### 3.5. Syntactic Space
* **Action:** Extraction of Dependency Distance, Syntactic Depth, and Root Status.
* **Methodological Decision:** Parsimonious parameterization based on integer counts (`np.int16`) from spaCy's dependency trees.
* **Justification:** Controls clausal integration dynamics and working memory load (Gibson's Integration Theory). These general syntactic structuring phenomena may covary with the appearance of main (finite) verbs and generate sharp BOLD signal fluctuations, thus requiring adequate control.

### 3.6. Reduced Lexical-Semantic Space (LSA)
* **Action:** Construction of a Latent Semantic Analysis (TF-IDF + Truncated SVD) reduced to 10 latent dimensions.
* **Methodological Decision 1 (Lemmas and Conversational Stopwords):** The global pipeline was trained using only lemmas. An empirically adjusted dictionary of stopwords was injected after an audit, filtering out conversational fillers (e.g., *uh, um, like*) and light verbs (e.g., *know, say, get*).
* **Justification 1:** Lemmatization ensures the model captures abstract content rather than inflectional surface forms. Manual filtering was critical, as the SVD algorithm initially interpreted hyper-frequent fillers as the "main topic" of the stories. Removing them revealed the true semantic axes of the narrative discourse.
* **Methodological Decision 2 (Semantic State Maintenance):** The resulting LSA vectors per clause (delimited by >0.65s pauses) were projected by holding their values temporally.
* **Justification 2:** Models how the brain updates and retains its semantic state sentence by sentence within the listener's working memory.

---

## 4. Interest Space Construction

### Finite Grammatical Tense Space
* **Action:** Extraction of internal realizations of Past (Regular/Irregular) and Non-Past (Marked/Unmarked).
* **Methodological Decision:** Detection supported by Penn Treebank tags and morphological heuristics (*-ed* suffix). **Absolute exclusion** of defective modals (MD) and ambiguous contractions (e.g., *'d*) was implemented.
* **Justification:** Validates the temporal contrast while protecting the "conceptual purity" of the evaluated phenomenon. By excluding elements that intertwine tense, aspect, and mood (TAM), it prevents the variance attributed to this space from being driven by modalizations irrelevant to the purely inflectional contrast. For the confirmatory analysis, these 4 dimensions are collapsed into two global domains: Past and Non-Past.

---

## 5. Temporal Modeling and Hemodynamic Transformation

To align the feature matrix with the brain capture resolution (fMRI), the 100 Hz matrices were temporally transformed and downsampled to a TR of 2.0 seconds.

* **Methodological Decision 1 (Double-Gamma Function):** A Double-Gamma HRF (SPM canonical parameters) was used instead of a Single-Gamma model.
* **Justification 1:** Benchmarking tests demonstrated that the computational cost was identical. The Double-Gamma offers biological superiority by accurately modeling the post-stimulus metabolic undershoot.
* **Methodological Decision 2 (FFT-Optimized Convolution):** Standard convolution was replaced by one based on the Fast Fourier Transform (`scipy.signal.fftconvolve`).
* **Justification 2:** Convolving temporally massive matrices (15-minute stories at 100 Hz against 32-second filters) using iterative methods presents an $O(N \times M)$ complexity. The FFT reduces this to $O(N \log N)$, exploiting the CPU's multi-core capacity and cutting computation time from hours to seconds.
* **Methodological Decision 3 (Anti-Aliasing Decimation):** Downsampling was executed using SciPy's `resample_poly` function.
* **Justification 3:** Aggressive reduction from 100 Hz to 0.5 Hz (TR=2s) introduces severe phase errors (aliasing). This method natively applies a polyphase FIR low-pass filter (anti-aliasing), protecting the frequency integrity of the activation peaks.

---

## 6. Predictive Architecture and Statistical Inference (Machine Learning)

Individual estimation (subject by subject) was performed using massive models over the entire cortex simultaneously (up to ~110,000 voxels per subject).

### 6.1. Hardware Decision (CPU vs GPU)
* **Justification:** Despite the availability of graphics acceleration, processing was strictly prioritized on the CPU (Intel i9, 16 threads, 64 GB RAM). A Banded Ridge Regression architecture with multiple spaces requires computing simultaneous dense matrices and multiple memory copies during cross-validation. Operating on a GPU with 4 GB of dedicated VRAM would inevitably result in *Out of Memory* crashes or *Memory Thrashing* (extreme bottlenecking due to constant PCIe bus transfers). The CPU fundamentally solved this via iterative block computation and low-level parallelization over NumPy.

### 6.2. Modeling Strategy
* **Methodological Decision 1 (Banded Ridge Regression):** Implemented using the `himalaya` library with independent L2 hyperparameterization (`alphas`) for each functional space.
* **Justification 1:** Given that the spaces have radically different dimensions and levels of collinearity, a global penalty (standard Ridge) would have allowed the largest space (e.g., phonological) to mathematically dominate the prediction. The banded variant with *random_search* calibrates the optimal weight of each neurocognitive domain separately.
* **Methodological Decision 2 (Strict Standardization - Z-score):** Means and variances of the spaces were learned exclusively from the training set and subsequently applied to the evaluation set. Furthermore, null imputation was forced on extra-cortical voxels (fMRI background) marked as `NaN` to avoid arithmetic collapses.
* **Justification 2:** Prevention of *Data Leakage*. Guarantees that the $\Delta R^2$ (Unique Predictive Variance) represents genuine out-of-sample generalization capacity, avoiding optimistic biases.

### 6.3. Inferential Statistical Validation
* **Action:** Empirical generation of a null distribution via *Circular Shift* and error control using *FDR* correction.
* **Justification:** Unlike a classic random permutation (shuffle), circular shifting destroys the stimulus-cortex association while preserving the inherent temporal autocorrelation of fMRI blood flow. The Benjamini-Hochberg correction (False Discovery Rate, FDR) restricts the emergence of false positives derived from performing tens of thousands of simultaneous contrasts (one per voxel).

---

## 7. Cortical Anatomical Visualization

To topographically inspect the model's statistical inference at the individual level, a 3D exploration interface was implemented using `pycortex`.

* **Methodological Decision (Evading internal API limits):** A system-level integration was designed to load anatomical transformations directly from the local *Filestore* (`.json`, `.ctm`), evading restrictions dependent on PyCortex API versions and generating freely accessible static web servers (WebGL/HTML).
* **Rendering Justification (Volume vs Vertex):** OpenNeuro fMRI data operate volumetrically, requiring the flattened $\Delta R^2$ to be projected using the `cortex.Volume` class with univariate heatmaps (e.g., `hot`), or using the `cortex.VolumeRGB` class for bivariate mapping (addition of light channels for Past and Non-Past). This ensures that cortical meshes are hydrated from fMRIPrep transformations without relying on 2D texture files susceptible to browser read errors. This provides the study with a highly precise, reproducible analytical and graphical resource for Open Science environments.