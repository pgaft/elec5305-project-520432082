# Guitar Onset Detection and Playing Technique Classification

**Author:** Patrick Gaft

**SID:** 520432082

**Course:** ELEC5305 - Audio Processing

## Project Overview

This project investigates how different parameter configurations influence the accuracy of spectral flux-based onset detection and how that detection accuracy correlates with downstream playing technique classification (e.g., Normal, Bend, Slide, Vibrato).

The system is modularised into three distinct pipelines:

1. **Pipeline 1 (Onset Optimization):** Performs a grid search over ```librosa``` spectral flux parameters (window size, hop length, thresholding) to maximize F-measure against the GuitarSet dataset.

2. **Pipeline 2 (Classification Baseline):** Extracts MFCC and spectral features to train an SVM classifier on the IDMT-SMT-Guitar dataset using ground-truth segmentation.

3. **Pipeline 3 (Joint Analysis):** Retrains the classifiers using detected onsets from Pipeline 1 to measure the impact of segmentation errors on classification performance.

## Installation & Requirements

This project requires ```Python 3.8+```.

### Dependencies

Install the required packages using ```pip```:

```
pip install librosa numpy pandas jams tqdm mir_eval scikit-learn matplotlib seaborn
```

### Dataset Setup

To run the experiments, you must download two datasets.

#### 1. GuitarSet

Used for optimizing onset detection parameters (Pipeline 1).

**Download Link:** [Zenodo Record 3371780](https://zenodo.org/records/3371780)

**Files required:**
- ```annotation.zip```
- ```audio_mono-mic.zip```

#### 2. IDMT-SMT-GUITAR V2

Used for playing technique classification (Pipeline 2 & 3).

Download Link: [Zenodo Record 7544110](https://zenodo.org/records/7544110)

File required: 
- ```IDMT-SMT-GUITAR_V2.zip```

### Configuration

Before running the code, you must update the file paths in the main script to point to your unzipped data directories.

Open ```main.py``` and edit the following lines at the top:

#### Path to the GuitarSet root folder (containing 'audio' and 'annotation' subfolders)
```GUITARSET_PATH = "/path/to/your/downloads/GuitarSet/"```

#### Path to the IDMT-SMT-GUITAR_V2 root folder
```IDMT_PATH = "/path/to/your/downloads/IDMT-SMT-GUITAR_V2/"```


## Usage

To run the full experiment chain (all 3 pipelines), simply execute the main script:
```
python main.py
```

- Note: The full pipeline involves a grid search and feature extraction over thousands of files. It takes *approximately 20 minutes* to run on a standard machine.

## Outputs

The script will generate console logs detailing the optimization progress and two visualization files in the root directory:

- ```confusion_matrix_baseline.png```: Visualizing the classification performance on ground-truth data.

- ```joint_analysis_plot.png```: plotting the relationship between Onset F-measure and Classification F1-score.