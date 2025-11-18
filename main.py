"""
Onset Detection and Playing Technique Classification for Guitar Recordings
"""

import librosa
import numpy as np
import pandas as pd
import jams
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Evaluation & ML
import mir_eval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

GUITARSET_PATH = "D:/5305-project-dataset/GuitarSet"
IDMT_PATH = "D:/5305-project-dataset/IDMT-SMT-GUITAR_V2"

ONSET_EVAL_WINDOW = 0.05
TARGET_SR = 22050
TECHNIQUES_TO_CLASSIFY = [
    'normal', 'bend', 'vibrato', 'slide', 'hammer-on', 'pull-off'
]

# Data Loading

def get_guitarset_files(data_dir):
    """Finds all audio and JAMS file pairs in GuitarSet.

    Handles GuitarSet naming like:
        audio_mono-mic/00_BN1-129-Eb_comp_mic.wav
        annotation/  00_BN1-129-Eb_comp.jams
    by stripping '_mic' / '_pickup' suffixes.
    """
    audio_dirs = []
    mic_dir = os.path.join(data_dir, "audio_mono-mic")
    pickup_dir = os.path.join(data_dir, "audio_mono-pickup")

    if os.path.isdir(mic_dir):
        audio_dirs.append(mic_dir)
    if os.path.isdir(pickup_dir):
        audio_dirs.append(pickup_dir)

    if not audio_dirs:
        print("No GuitarSet audio directories found (audio_mono-mic / audio_mono-pickup).")
        return []

    # Collect audio files from all available dirs
    audio_files = []
    for ad in audio_dirs:
        audio_files.extend(glob.glob(os.path.join(ad, "*.wav")))
    audio_files = sorted(audio_files)

    # JAMS files
    jams_files = sorted(glob.glob(os.path.join(data_dir, "annotation", "*.jams")))

    # Map JAMS by base name (without extension)
    jams_map = {os.path.basename(f).replace(".jams", ""): f for f in jams_files}

    file_pairs = []
    for audio_file in audio_files:
        base_name = os.path.basename(audio_file).replace(".wav", "")

        for suffix in ["_mic", "_pickup"]:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        if base_name in jams_map:
            file_pairs.append((audio_file, jams_map[base_name]))

    print(f"Found {len(file_pairs)} audio/JAMS pairs in GuitarSet.")
    return file_pairs

def load_guitarset_onsets(jams_file):
    """Loads ground-truth onsets from a GuitarSet JAMS file."""
    jam = jams.load(jams_file)
    onset_data = jam.search(namespace='note_midi')
    onsets = []
    if onset_data:
        for note in onset_data[0].data:
            onsets.append(note.time)
    return np.array(sorted(list(set(onsets))))

def load_idmt_annotations(annotation_file):
    """
    Load note-level annotations from an IDMT-SMT-GUITAR XML file
    (datasets 1–3 format as described in SMT_GUITAR_dataset_description.pdf).

    We use:
      - onsetSec / offsetSec   -> note boundaries (seconds)
      - expressionStyle        -> expression class (NO, BE, SL, VI, ...)

    Mapping from IDMT codes to our technique labels:
      NO -> 'normal'
      BE -> 'bend'
      SL -> 'slide'
      VI -> 'vibrato'

    Other expression styles (HA, DN, FL, ST, TR, etc.) are ignored.
    """
    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        ground_truth = []

        # Iterate over each <event> in <transcription>
        # (".//event" is robust to slight structural variations)
        for event in root.findall(".//event"):
            onset_el = event.find("onsetSec")
            offset_el = event.find("offsetSec")
            expr_el = event.find("expressionStyle")

            # Need onset, offset, and expression style
            if onset_el is None or offset_el is None or expr_el is None:
                continue

            try:
                start_time = float(onset_el.text)
                end_time = float(offset_el.text)
            except (TypeError, ValueError):
                continue

            if end_time <= start_time:
                continue

            # Expression style is coded as NO, BE, SL, VI, ...
            expr_code = (expr_el.text or "").strip().upper()
            expr_map = {
                "NO": "normal",
                "BE": "bend",
                "SL": "slide",
                "VI": "vibrato",
            }

            label = expr_map.get(expr_code, None)
            if label is None:
                # Ignore HA (harmonics), DN (dead-notes), etc.
                continue

            # Only keep techniques we care about
            if label in TECHNIQUES_TO_CLASSIFY:
                ground_truth.append({
                    "start": start_time,
                    "end": end_time,
                    "label": label,
                })

        # Debug print to see which XMLs truly have no usable notes
        if not ground_truth:
            print(f"Warning: no usable note events found in {os.path.basename(annotation_file)}")

        return ground_truth

    except ET.ParseError:
        print(f"Warning: Failed to parse XML: {annotation_file}")
        return []
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return []

def load_audio(audio_file, sr=TARGET_SR):
    """Loads and resamples audio."""
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    return y

# Pipeline 1: Onset Detection

def _adaptive_mean_peak_pick(onset_env, delta, window=3):
    """Adaptive mean threshold peak picking (Rosão et al., 2012)."""
    threshold = np.zeros_like(onset_env)
    for i in range(len(onset_env)):
        start = max(0, i - window)
        end = min(len(onset_env), i + window + 1)
        threshold[i] = np.mean(onset_env[start:end]) + delta
    
    peaks = []
    for i in range(1, len(onset_env) - 1):
        if (onset_env[i] > onset_env[i-1] and 
            onset_env[i] > onset_env[i+1] and 
            onset_env[i] > threshold[i]):
            peaks.append(i)
    return np.array(peaks)

def detect_onsets(y, sr=22050, n_fft=1024, hop_length=512,
                  threshold_method='adaptive_median', delta=0.07):
    """
    Spectral flux onset detection with configurable parameter
    """
    # Compute onset strength (spectral flux)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        aggregate=np.median
    )
    
    onset_frames = []
    if threshold_method == 'adaptive_median':
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
            units='frames', delta=delta, wait=1,
            pre_avg=1, post_avg=1, pre_max=1, post_max=1
        )
    elif threshold_method == 'adaptive_mean':
        onset_frames = _adaptive_mean_peak_pick(onset_env, delta)
    else:  # fixed
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
            units='frames', delta=delta, wait=0
        )
        
    onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onsets

def evaluate_onset_detection(detected_onsets, reference_onsets, window=0.05):
    """
    Evaluate using mir_eval with 50ms tolerance
    """
    precision, recall, f_measure = mir_eval.onset.f_measure(
        reference_onsets, detected_onsets, window=window
    )
    return {'precision': precision, 'recall': recall, 'f_measure': f_measure}

def run_onset_detection_experiment(file_pairs):
    """
    Runs the systematic evaluation of onset detection parameters
    based on the user's 108-configuration grid.
    """
    print("\nRunning Pipeline 1: Onset Detection Experiment")
    
    # Parameter grid: 3 window sizes × 3 hop lengths × 3 threshold methods × 4 delta values = 108
    param_grid = {
        'n_fft': [512, 1024, 2048],
        'hop_length': [128, 256, 512],
        'threshold_method': ['adaptive_median', 'adaptive_mean', 'fixed'],
        'delta': [0.05, 0.07, 0.1, 0.2] # 4 delta values
    }
    
    results = []
    pbar = tqdm(total=len(file_pairs) * 3 * 3 * 3 * 4, desc="Evaluating Onset Configs")

    for audio_file, annot_file in file_pairs:
        y = load_audio(audio_file)
        gt_onsets = load_guitarset_onsets(annot_file)
        
        if len(gt_onsets) == 0:
            pbar.update(3 * 3 * 3 * 4)
            continue
            
        for n_fft in param_grid['n_fft']:
            for hop_length in param_grid['hop_length']:
                if hop_length >= n_fft: # Skip invalid configs
                    pbar.update(3 * 4)
                    continue
                
                for threshold_method in param_grid['threshold_method']:
                    for delta in param_grid['delta']:
                        params = {
                            'n_fft': n_fft,
                            'hop_length': hop_length,
                            'threshold_method': threshold_method,
                            'delta': delta
                        }
                        
                        try:
                            detected_onsets = detect_onsets(y, TARGET_SR, **params)
                            eval_scores = evaluate_onset_detection(detected_onsets, gt_onsets)
                            
                            results.append({
                                'params': params,
                                'f_measure': eval_scores['f_measure'],
                                'precision': eval_scores['precision'],
                                'recall': eval_scores['recall']
                            })
                        except Exception as e:
                            print(f"Error in onset detection with params {params}: {e}")
                            results.append({
                                'params': params,
                                'f_measure': 0, 'precision': 0, 'recall': 0
                            })
                        pbar.update(1)

    pbar.close()
    
    # Aggregate results
    df_results = pd.DataFrame(results)

    # Convert params dict to a string key for grouping
    df_results['params_str'] = df_results['params'].apply(str)

    # Only average numeric metric columns, not the 'params' dict column
    metric_cols = ['f_measure', 'precision', 'recall']
    agg_results = (
        df_results
        .groupby('params_str')[metric_cols]
        .mean()
        .sort_values(by='f_measure', ascending=False)
    )

    
    print("\n--- Onset Detection Results (Top 5) ---")
    print(agg_results[['f_measure', 'precision', 'recall']].head())
    
    # Store full (non-aggregated) results for joint analysis
    # We group by params and get the mean scores
    full_agg_results = df_results.groupby('params_str').agg({
        'params': 'first', # Get the actual dict back
        'f_measure': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index(drop=True).sort_values(by='f_measure', ascending=False)

    best_params_str = agg_results.index[0]
    best_f1 = agg_results.iloc[0]['f_measure']
    print(f"\nBest configuration: {best_params_str} (F-measure: {best_f1:.4f})")
        
    return full_agg_results

# Pipeline 2: Playing Technique Classification

def extract_features(y_segment, sr=TARGET_SR, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCCs + delta + delta-delta + spectral centroid + rolloff + ZCR
    and aggregate by mean and std over time.

    Short segments are zero-padded so that we always have at least 9 frames,
    which avoids librosa.feature.delta width errors.
    """
    # Ensure minimum length for at least 9 frames
    # For STFT-based features, number of frames ~= 1 + floor((len - n_fft) / hop_length)
    # We want at least 9 frames, so we pad if needed.
    min_frames = 9
    min_len = n_fft + (min_frames - 1) * hop_length
    if len(y_segment) < min_len:
        pad_width = min_len - len(y_segment)
        y_segment = np.pad(y_segment, (0, pad_width), mode="constant")

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=y_segment,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # Now mfcc.shape[1] >= 9, so default width=9 is safe
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=y_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=0.85,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y_segment,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    def agg(feat):
        # Aggregate over time: mean and std for each coefficient
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    feats = [
        agg(mfcc),
        agg(mfcc_delta),
        agg(mfcc_delta2),
        agg(centroid),
        agg(rolloff),
        agg(zcr),
    ]

    return np.concatenate(feats)

def train_technique_classifier(X_train, y_train):
    """
    Train SVM classifier with grid search.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Grid search for optimal C and gamma
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }
        
    svm = SVC(cache_size=500) # Add cache for speed
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
    )
    print("Running GridSearchCV for SVM (C and gamma)...")
    grid_search.fit(X_train_scaled, y_train)
        
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1: {grid_search.best_score_:.4f}")
        
    return grid_search.best_estimator_, scaler

def evaluate_technique_classification(y_true, y_pred, class_names):
    """
    Evaluate multi-class technique classification.

    y_true, y_pred are label-encoded integers.
    class_names is an array like le.classes_ (index -> string label).

    We only evaluate and name the classes that actually appear in y_true/y_pred,
    which avoids mismatches when some techniques have no examples in the dataset
    (e.g., hammer-on / pull-off in IDMT).
    """
    # Which label indices actually appear?
    labels_used = np.unique(np.concatenate([y_true, y_pred]))

    # Map those indices to their corresponding names
    effective_class_names = [class_names[i] for i in labels_used]

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Use the same 'labels_used' ordering for both confusion matrix and report
    cm = confusion_matrix(y_true, y_pred, labels=labels_used)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_used,
        target_names=effective_class_names,
        zero_division=0,
    )

    return {
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "report": report,
        "labels": np.array(effective_class_names),
    }

def get_audio_annotation_pairs(idmt_audio_dir):
    """Finds audio/annotation file pairs for IDMT."""

    audio_files = glob.glob(os.path.join(idmt_audio_dir, "**", "*.wav"), recursive=True)
    file_pairs = []

    for audio_file in audio_files:
        annot_path = audio_file.replace(os.sep + "audio" + os.sep, os.sep + "annotation" + os.sep)

        annot_file = os.path.splitext(annot_path)[0] + ".xml" # <-- We are looking for .xml

        if os.path.exists(annot_file):
            file_pairs.append((audio_file, annot_file))

    print(f"Found {len(file_pairs)} audio/annotation pairs in IDMT.")
    return file_pairs

def process_dataset_for_classification(file_pairs, use_ground_truth_onsets=True, onset_params=None):
    """
    Loads IDMT audio, segments it, extracts features, and returns X, y.
    This function is adapted to be flexible for both GT and detected onsets.
    """
    X_data = []
    y_data = []

    desc = "Extracting Features (GT Onsets)" if use_ground_truth_onsets else "Extracting Features (Detected Onsets)"

    for audio_file, annot_file in tqdm(file_pairs, desc=desc):
        annotations = load_idmt_annotations(annot_file)
        if not annotations:
            continue  # skip files with no usable notes

        y_audio = load_audio(audio_file, sr=TARGET_SR)

        if use_ground_truth_onsets:
            # --- Ground-Truth Segmentation ---
            local_count = 0
            for note in annotations:
                start_sample = librosa.time_to_samples(note["start"], sr=TARGET_SR)
                end_sample = librosa.time_to_samples(note["end"], sr=TARGET_SR)

                if end_sample <= start_sample:
                    continue

                y_segment = y_audio[start_sample:end_sample]
                features = extract_features(y_segment, TARGET_SR)
                X_data.append(features)
                y_data.append(note["label"])
                local_count += 1

            if local_count == 0:
                print(f"Warning: {os.path.basename(audio_file)} had annotations but produced no segments.")
        else: 
            # --- Detected Onset Segmentation ---
            if onset_params is None:
                raise ValueError("`onset_params` must be provided.")
            
            detected_onsets = detect_onsets(y_audio, TARGET_SR, **onset_params)
            if len(detected_onsets) == 0:
                continue

            gt_starts = np.array([n['start'] for n in annotations])
            gt_ends = np.array([n['end'] for n in annotations])
            
            for i in range(len(detected_onsets)):
                t_onset = detected_onsets[i]
                
                # Find ground-truth note that contains this onset
                matches = np.where((gt_starts <= t_onset) & (t_onset < gt_ends))[0]
                
                if len(matches) == 1:
                    gt_index = matches[0]
                    gt_label = annotations[gt_index]['label']
                    
                    # Segment: from this onset to the *next* onset
                    start_sample = librosa.time_to_samples(t_onset, sr=TARGET_SR)
                    end_sample = None
                    if i < len(detected_onsets) - 1:
                        end_sample = librosa.time_to_samples(detected_onsets[i+1], sr=TARGET_SR)
                    else:
                        # Use end of the ground-truth note as final boundary
                        end_sample = librosa.time_to_samples(annotations[gt_index]['end'], sr=TARGET_SR)
                    
                    y_segment = y_audio[start_sample:end_sample]
                    features = extract_features(y_segment, TARGET_SR)
                    X_data.append(features)
                    y_data.append(gt_label)

    # Convert to NumPy arrays and encode labels
    le = LabelEncoder().fit(TECHNIQUES_TO_CLASSIFY)
    y_encoded = le.transform(y_data)
    
    X_data = np.array(X_data)
    X_data = np.nan_to_num(X_data, copy=False)

    print(f"Processed {len(X_data)} feature vectors.")
    return X_data, y_encoded, le, (len(X_data) > 0)

def run_classification_experiment(X_train, y_train, X_test, y_test, le):
    """
    Trains and evaluates the SVM classifier (Baseline).
    """
    print("\nRunning Pipeline 2: Technique Classification Experiment (Baseline)")
    
    # 1. Train the model using the user's function
    baseline_model, scaler = train_technique_classifier(X_train, y_train)
    
    # 2. Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Evaluate on the test set
    y_pred = baseline_model.predict(X_test_scaled)
    
    # 4. Report results using the user's function
    print("\nBaseline Classification Results (on Test Set)")
    results = evaluate_technique_classification(y_test, y_pred, le.classes_)
    print(results['report'])
    print(f"Baseline Macro F1-score: {results['macro_f1']:.4f}")

    
    # Plot confusion matrix
    cm = results['confusion_matrix']
    labels = results['labels']

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )

    plt.title("Confusion Matrix - Baseline (GT Onsets)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_baseline.png")
    print("Saved baseline confusion matrix to 'confusion_matrix_baseline.png'")
    
    return results['macro_f1']

# Pipeline 3: Joint Analysis

def run_joint_analysis(train_files, test_files, df_onset_results, le):
    """
    Quantifies how onset detection errors affect classification
    by retraining the model on mis-segmented data.
    """
    print("\nRunning Pipeline 3: Joint Analysis (Retraining Protocol)")
    
    # 1. Define target onset F-measures
    target_f_measures = [0.6, 0.7, 0.8]
    selected_configs = []
    
    for f_target in target_f_measures:
        # Find the config with F-measure *closest* to the target
        closest_config_idx = (df_onset_results['f_measure'] - f_target).abs().idxmin()
        config_data = df_onset_results.loc[closest_config_idx]
        selected_configs.append(config_data)
        print(f"Selected config for target F1 ~{f_target}: "
              f"(Actual F1: {config_data['f_measure']:.4f}, "
              f"Params: {config_data['params']})")
    
    joint_results = []

    # 2. For each selected onset configuration:
    for config_data in selected_configs:
        onset_f1 = config_data['f_measure']
        onset_params = config_data['params']
        
        print(f"\n--- Retraining classifier for Onset F1: {onset_f1:.4f} ---")
        
        # 3. Segment IDMT *training* audio using detected onsets
        X_mis_train, y_mis_train, _, success_train = process_dataset_for_classification(
            train_files, 
            use_ground_truth_onsets=False, 
            onset_params=onset_params
        )
        
        # 4. Segment IDMT *test* audio using detected onsets
        X_mis_test, y_mis_test, _, success_test = process_dataset_for_classification(
            test_files, 
            use_ground_truth_onsets=False, 
            onset_params=onset_params
        )
        
        if not success_train or not success_test:
            print("Skipping config: failed to extract features for train or test set.")
            continue
            
        # 5. Retrain technique classifier on this mis-segmented data
        print(f"Retraining model on {len(X_mis_train)} mis-segmented training samples...")
        mis_segmented_model, mis_scaler = train_technique_classifier(X_mis_train, y_mis_train)
        
        # 6. Evaluate the *new* model on the mis-segmented *test* data
        X_mis_test_scaled = mis_scaler.transform(X_mis_test)
        y_pred = mis_segmented_model.predict(X_mis_test_scaled)
        
        # 7. Compare technique classification F1 to baseline
        results = evaluate_technique_classification(y_mis_test, y_pred, le.classes_)
        tech_f1_macro = results['macro_f1']

        print(f"Technique Classification F1-score (for Onset F1 {onset_f1:.4f}): {tech_f1_macro:.4f}")
        joint_results.append({
            'onset_f_measure': onset_f1,
            'technique_f1_macro': tech_f1_macro
        })


    # 8. Plot relationship
    print("\n--- Joint Analysis Summary ---")
    df_joint = pd.DataFrame(joint_results)
    print(df_joint)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_joint, x='onset_f_measure', y='technique_f1_macro', s=100)
    sns.lineplot(data=df_joint, x='onset_f_measure', y='technique_f1_macro', marker='o')
    plt.title("Impact of Onset Detection F-measure on Technique Classification F1")
    plt.xlabel("Onset Detection F-measure")
    plt.ylabel("Technique Classification F1-macro (Retrained Model)")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig("joint_analysis_plot.png")
    print("Saved joint analysis plot to 'joint_analysis_plot.png'")

if __name__ == "__main__":
    
    # Run Pipeline 1: Onset Detection
    df_onset_results = None
    try:
        guitarset_files = get_guitarset_files(GUITARSET_PATH)
        if not guitarset_files:
            raise FileNotFoundError("GuitarSet files not found. Check GUITARSET_PATH.")
        
        df_onset_results = run_onset_detection_experiment(guitarset_files)
    
    except FileNotFoundError as e:
        print(e)
        print("Skipping Pipeline 1. Joint analysis will not be possible.")
    
    # Run Pipeline 2 & 3: Classification
    try:
        IDMT_AUDIO_DIR = IDMT_PATH # MODIFY AS NEEDED
        idmt_files = get_audio_annotation_pairs(IDMT_AUDIO_DIR)
        if not idmt_files:
            raise FileNotFoundError("IDMT files not found. Check IDMT_PATH.")
            
        # 1. Split IDMT file list (70% train, 30% test)
        train_files, test_files = train_test_split(idmt_files, test_size=0.3, random_state=42)
        
        # 2. Process Baseline (Ground Truth) data
        X_gt_train, y_gt_train, le, success_train = process_dataset_for_classification(
            train_files, use_ground_truth_onsets=True
        )
        X_gt_test, y_gt_test, _, success_test = process_dataset_for_classification(
            test_files, use_ground_truth_onsets=True
        )
        
        if not success_train or not success_test:
            raise RuntimeError("Failed to load GT data from IDMT. Check paths/annotations.")

        # 3. Run Baseline Classification (Pipeline 2)
        run_classification_experiment(X_gt_train, y_gt_train, X_gt_test, y_gt_test, le)
        
        # 4. Run Joint Analysis (Pipeline 3)
        if df_onset_results is not None:
            run_joint_analysis(train_files, test_files, df_onset_results, le)
        else:
            print("\nSkipping Pipeline 3 (Joint Analysis) because Pipeline 1 failed.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nFailed to run classification pipelines: {e}")
        print("Please check your IDMT_PATH and annotation loading function (`load_idmt_annotations`).")

    print("\nProject Execution Complete")