# EEGNet Reproduction Study – BCI Course Project (Option C, Track C1)

**Project type:** Individual course project  
**Base paper:** Lawhern et al. (2018), *EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces*  
**Dataset:** BCI Competition IV – Dataset 2a  
**Frameworks:** TensorFlow/Keras and MNE-Python

## 1. Project overview

This project implements an EEGNet-8,2 model for four-class motor-imagery classification using the SMR recordings in BCI Competition IV Dataset 2a. The work is based on the EEGNet paper by Lawhern et al. and evaluates the model across nine subjects.

The implementation uses the original GDF recordings and the official evaluation-session labels. It includes EEG preprocessing, EEGNet model construction, cross-subject training and evaluation, result tables, and model-interpretation visualisations.

This is an independent implementation guided by the paper. It is **not an exact reproduction of every preprocessing and interpretation detail** in the original study. The implementation differences are listed in the Limitations section below.

## 2. Base paper and dataset

### Base paper

Lawhern, V. J., et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces.** *Journal of Neural Engineering*.

- Journal article: <https://doi.org/10.1088/1741-2552/aace8c>
- Open full-text PDF: <https://arxiv.org/pdf/1611.08024>
- Authors' EEGNet implementation: <https://github.com/vlawhern/arl-eegmodels>

### Dataset

The project uses **BCI Competition IV, Dataset 2a**.

- Official dataset page: <https://www.bbci.de/competition/iv/download/>
- Dataset description: <https://www.bbci.de/competition/iv/desc_2a.pdf>
- Official evaluation labels: <https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip>

Dataset characteristics used in this project:

- 9 subjects (A01–A09)
- 22 EEG channels
- 4 motor-imagery classes: left hand, right hand, feet, and tongue
- 250 Hz original sampling frequency, resampled to 128 Hz
- 288 trials in each training session and 288 trials in each evaluation session
- 72 trials per class in each session

The GDF recordings are named `A01T.gdf` to `A09T.gdf` and `A01E.gdf` to `A09E.gdf`. Evaluation labels are loaded from the official subject-specific MAT files.

## 3. Processing and modelling pipeline

The notebook follows this sequence:

1. **Load data:** Read the original GDF recordings and official evaluation labels.
2. **Channel handling:** Map the GDF channel identifiers to the standard 22 BCI IV-2a electrode names and arrange them in a consistent order.
3. **Resample:** Convert the signals from 250 Hz to 128 Hz.
4. **Filter:** Apply a 4–40 Hz zero-phase FIR band-pass filter.
5. **Epoch:** Extract two-second epochs from 0.5 to 2.5 seconds after the cue.
6. **Prepare labels:** Use training-session event codes 769–772 and the official labels for evaluation-session cue events.
7. **Train EEGNet:** Train the EEGNet-8,2 network using the training and validation subjects for each split.
8. **Evaluate and interpret:** Calculate accuracy and Cohen's kappa, summarise subject-wise performance, and visualise learned filters and input-gradient magnitudes.

Signals are converted from volts to microvolts before model training.

## 4. EEGNet-8,2 architecture

The model uses the compact EEGNet architecture with the following settings:

| Parameter | Setting |
|---|---:|
| Temporal filters (F1) | 8 |
| Depth multiplier (D) | 2 |
| Pointwise filters (F2) | 16 |
| Temporal convolution kernel | 32 samples |
| Dropout rate | 0.25 |
| Output classes | 4 |
| Trainable parameters | 1,716 |
| Total Keras parameters | 1,796 |

The network contains temporal convolution, depthwise spatial convolution, separable convolution, and a four-class softmax classifier.

## 5. Cross-subject evaluation protocol

For each held-out target subject:

- Five of the other eight subjects are selected for training.
- The remaining three subjects are used for validation.
- The target subject's evaluation session is used only for testing.
- The five-subject/three-subject split is repeated ten times.

This gives **9 target subjects × 10 repetitions = 90 evaluation runs**. The target subject's training session is not used to train or validate the model in this protocol.

Training settings in this implementation:

| Setting | Value |
|---|---:|
| Maximum epochs | 500 |
| Batch size | 32 |
| Early-stopping patience | 50 epochs |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Random seed | 42 |

Batch size and early-stopping patience are implementation choices; they are not stated as exact values in the paper's SMR description.

## 6. Experimental results

The completed run contains 90 evaluation results.

| Metric | Result |
|---|---:|
| Number of evaluation runs | 90 |
| Mean accuracy | **37.7392%** |
| Standard deviation across runs | 13.6181 percentage points |
| Standard error of the mean | 1.4355 percentage points |
| Mean Cohen's kappa | 0.169856 |
| Four-class chance-level accuracy | 25.00% |

### Subject-wise mean accuracy

| Held-out subject | Mean accuracy |
|---|---:|
| A01 | 53.19% |
| A02 | 27.95% |
| A03 | 53.68% |
| A04 | 33.30% |
| A05 | 26.28% |
| A06 | 29.51% |
| A07 | 28.92% |
| A08 | 50.76% |
| A09 | 36.04% |

The results show variation across subjects. The overall mean is above the four-class chance level, but performance is not uniform across participants.

The notebook also contains a comparison table using values visually estimated from the paper's figure. Those values are approximate graphical readings, not exact numerical values published in a result table. They should be treated as context rather than a precise statistical comparison.

## 7. Reproduction notes and limitations

The results should be described as an **approximate reproduction / independent implementation**, not an exact reproduction of the complete original pipeline.

1. **Filtering:** This notebook uses a zero-phase FIR filter in MNE for the 4–40 Hz passband. This differs from the causal third-order Butterworth filtering used in the referenced preprocessing procedure.
2. **Standardisation:** The exponential moving standardisation procedure described in the referenced preprocessing method was not implemented.
3. **Artifact markers:** Marker 1023 is counted and reported in the quality summary. Epoch extraction uses `reject_by_annotation=False`, so artifact-marked trials are retained.
4. **Training settings:** Batch size 32 and early-stopping patience 50 are implementation choices. The paper's SMR description does not specify these exact values.
5. **Interpretability:** Temporal and spatial filter weights are visualised. Input-gradient magnitude is used as a supplementary single-trial interpretation method; it is not the DeepLIFT method used in the paper.
6. **Numerical differences:** Hardware, software versions, random initialisation, and preprocessing implementation can affect the measured accuracy.

These differences are documented so that the reported results are interpreted transparently.



Only include files that are actually present in the repository. The original EEG recordings and official label archive should be obtained from their official source; do not upload restricted or unnecessarily large raw data files to the repository.

## 8. How to run the notebook

The notebook is prepared for **Kaggle Notebooks**.

1. Download or import `EEGNet_C1_Final_Reproducible.ipynb` into Kaggle.
2. Use **Add Input** to attach a dataset containing the original BCI IV-2a GDF recordings, if available.
3. Enable Internet if the notebook needs to download the official GDF archive or evaluation labels.
4. Select a GPU accelerator for the full experiment.
5. Run the notebook cells from top to bottom.
6. Find generated tables and model artifacts under `/kaggle/working/eegnet_c1_reproduction/` and the final export under `/kaggle/working/EEGNet_C1_Final_Export/`.

**Important:** Running the notebook from the beginning trains the 90 evaluation runs again and may take considerable time. The previously completed run's CSV results should be kept separately as the record of that experiment. The corrected notebook source has not been runtime-validated end-to-end after the final channel-mapping edits.

## 9. Software environment

Versions observed during the completed Kaggle run:

- Python 3.13.16
- TensorFlow 2.21.0
- MNE 1.13.2

The notebook also uses NumPy, pandas, SciPy, scikit-learn, Matplotlib, and Requests. Exact versions for every supporting library were not recorded in the saved experiment summary.

## 11. Acknowledgements

The project uses the publicly released BCI Competition IV Dataset 2a and builds on the EEGNet architecture proposed by Lawhern et al. The original dataset and research paper are cited above.
