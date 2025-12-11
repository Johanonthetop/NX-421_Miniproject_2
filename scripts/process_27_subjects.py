#!/usr/bin/env python3
"""
Process multiple subject .mat files (up to 27) using the same pipeline
from the notebook: feature extraction (MAV, STD), train/test split, scaling,
train GradientBoostingClassifier and report accuracy & macro-F1 per subject.
Saves aggregated metrics to `results.csv` in the working directory.

Usage:
  python scripts/process_27_subjects.py --data-dir s2 --limit 27
  python scripts/process_27_subjects.py --data-dir s2 --limit 27 --grid

The `--grid` flag runs GridSearchCV per subject (slower).
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import convolve1d
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


def build_dataset_from_ninapro(emg, stimulus, repetition, features):
    # Number of stimuli/repetitions excluding resting/zero indices
    n_stimuli = np.unique(stimulus).size - 1
    n_repetitions = np.unique(repetition).size - 1
    n_samples = int(n_stimuli * n_repetitions)
    n_channels = emg.shape[1]
    n_features = n_channels * len(features)

    dataset = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    sample_idx = 0

    for i in range(n_stimuli):
        for j in range(n_repetitions):
            labels[sample_idx] = i + 1
            selected_tsteps = np.logical_and(stimulus == i + 1, repetition == j + 1).squeeze()
            current_feature_index = 0
            for feature in features:
                vals = feature(emg[selected_tsteps, :])
                dataset[sample_idx, current_feature_index:current_feature_index + n_channels] = vals
                current_feature_index += n_channels
            sample_idx += 1

    return dataset, labels


# Feature definitions
mav = lambda x: np.mean(np.abs(x), axis=0)
std = lambda x: np.std(x, axis=0)


def process_subject(mat_path, features, use_grid=False):
    data = loadmat(mat_path)
    # expected keys in the provided data (from the notebook)
    emg = data.get('emg')
    stimulus = data.get('restimulus') or data.get('stimulus')
    repetition = data.get('rerepetition') or data.get('repetition')

    if emg is None or stimulus is None or repetition is None:
        raise ValueError(f"Missing expected variables in {mat_path}. Found keys: {list(data.keys())}")

    dataset, labels = build_dataset_from_ninapro(emg, stimulus, repetition, features)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=0.4, stratify=labels, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=0)

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)
    X_test_z = scaler.transform(X_test)

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train_z, y_train)

    y_test_pred = gb.predict(X_test_z)
    baseline_acc = accuracy_score(y_test, y_test_pred)
    baseline_f1 = f1_score(y_test, y_test_pred, average='macro')

    best_params = None
    best_cv = None
    if use_grid:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4],
            'subsample': [0.7, 1.0],
            'max_features': ['sqrt', None],
        }
        grid = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        grid.fit(X_train_z, y_train)
        best = grid.best_estimator_
        best_params = grid.best_params_
        best_cv = grid.best_score_
        y_test_pred_best = best.predict(X_test_z)
        best_acc = accuracy_score(y_test, y_test_pred_best)
        best_f1 = f1_score(y_test, y_test_pred_best, average='macro')
    else:
        best_acc = baseline_acc
        best_f1 = baseline_f1

    return {
        'baseline_acc': float(baseline_acc),
        'baseline_f1': float(baseline_f1),
        'best_acc': float(best_acc),
        'best_f1': float(best_f1),
        'best_params': best_params,
        'best_cv': best_cv,
    }


def main():
    parser = argparse.ArgumentParser(description='Process .mat subjects and compute metrics')
    parser.add_argument('--data-dir', type=str, default='s2', help='Directory containing subject .mat files')
    parser.add_argument('--limit', type=int, default=27, help='Maximum number of subjects to process')
    parser.add_argument('--grid', action='store_true', help='Run GridSearchCV per subject (slow)')
    parser.add_argument('--out-csv', type=str, default='results.csv', help='Output CSV filename')

    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.data_dir) if f.lower().endswith('.mat')])
    if len(files) == 0:
        raise SystemExit(f"No .mat files found in {args.data_dir}")

    files = files[:args.limit]

    results = []
    for idx, fname in enumerate(files, start=1):
        path = os.path.join(args.data_dir, fname)
        print(f"[{idx}/{len(files)}] Processing {fname}...")
        try:
            res = process_subject(path, features=[mav, std], use_grid=args.grid)
            res_row = {
                'subject_file': fname,
                'baseline_acc': res['baseline_acc'],
                'baseline_f1': res['baseline_f1'],
                'best_acc': res['best_acc'],
                'best_f1': res['best_f1'],
                'best_params': json.dumps(res['best_params']) if res['best_params'] is not None else None,
                'best_cv': res['best_cv']
            }
            results.append(res_row)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            results.append({'subject_file': fname, 'error': str(e)})

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved results to {args.out_csv}")


if __name__ == '__main__':
    main()
