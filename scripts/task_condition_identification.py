#!/usr/bin/env python3
# this script is only for trial-level GLM outputs
"""
Task-condition identification & grouping for **trial-level** GLM outputs.

This adapts your prior (condition-based) script to the current trial-level
GLM data layout you generate per run:
  - HDF5:  <...>/<sub>/<ses>/func/<base>_betas.h5  with dataset `betas` of shape (P x K_trials)
  - CSV:   sibling <base>_design.csv with one row per trial regressor (in column order),
           including at least: [trialNumber, stim_order, location, object, category, is_correct, type]

What it does
------------
• For each run, it reads the **trial-level** `betas` and the matching design CSV.
• It selects **encoding** rows, then groups them within a trial according to task-specific
  consecutive-stimulus pairs (e.g., ctxdm: (1,2), (2,3); interdms: (1,2), (2,3), (3,4); etc.).
• For each pair, it builds a condition key (e.g., "ctxdm_locL_objA*locR_objB") and appends the
  voxel beta vector for the **first** stimulus in the pair (keeps behavior compatible with your
  earlier script; see `PAIR_SELECTION` flag to change to 'mean' if desired).
• Aggregates across all runs into a dict: {condition_key: [beta_vec, beta_vec, ...]}.
• Saves the dictionary as a pickle and writes a small repetition-count report + histogram.

Differences vs old script
-------------------------
• No `design_tsv` lookup is required anymore; we parse `task` from the filename.
• We map betas columns to trials directly via the design CSV row order (which matches the
  order the trial regressors were appended when you built X). This removes the 2*N == K check.
• Fixed `determine_feature_type` bug and made the task/trial sanity checks softer (warnings only).

Default paths
-------------
  --glm_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas
  --out_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas

Outputs
-------
  <out_root>/grouped_betas/task_relevant_only/<subj>_task_condition_betas.pkl
  <out_root>/results/repetition_count/task_relevant_only/all_consecutive_stim_pairs/<subj>_repetition_distribution.png

Usage
-----
python task_condition_identification.py \
  --subj sub-01 \
  --glm_root /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas \
  --tasks ctxdm interdms nback \
  --correct_only
"""
from __future__ import annotations

import os
import re
import h5py
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
PAIR_SELECTION = "first"  # {'first', 'mean'} — which encoding stimulus beta to collect per pair

# -----------------------
# Helpers
# -----------------------

def parse_run_tokens(beta_file: Path):
    """Extract (subj, ses, task, run, base_stem) from a betas filename.
    Expects pattern like: sub-01_ses-003_task-ctxdm_run-01_betas.h5
    """
    stem = beta_file.stem  # remove .h5
    # strip trailing _betas
    if stem.endswith("_betas"):
        base = stem[:-6]
    else:
        base = stem
    # tokens are separated by '_'
    parts = base.split('_')
    subj = next((p for p in parts if p.startswith('sub-')), 'sub-unknown')
    ses  = next((p for p in parts if p.startswith('ses-')), 'ses-unknown')
    task = next((p for p in parts if p.startswith('task-')), 'task-unknown')
    run  = next((p for p in parts if p.startswith('run-')), 'run-unknown')
    task_name = task.split('-', 1)[1] if '-' in task else task
    return subj, ses, task_name, run, base


def determine_task_type(task_name: str):
    """Return list of (stim_order_i, stim_order_j) pairs to consider per trial for a task."""
    t = task_name.lower()
    if t.startswith("interdms"):
        return [(1, 2), (2, 3), (3, 4)]
    elif t.startswith("ctxdm"):
        return [(1, 2), (2, 3)]
    elif t.startswith("nback") or t.startswith("1back"):
        # default: pairs of consecutive items in a 6-item sequence (adjust if needed)
        return [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    else:
        # sensible default: consecutive pairs (1,2), (2,3)
        return [(1, 2), (2, 3)]


def determine_feature_type(task_name: str):
    """Which features define a condition key for the task.
    Returns a list subset of {"location", "object"}.
    """
    tn = task_name.lower()
    if ("obj" in tn) or ("ctg" in tn):
        return ["object"]
    elif ("loc" in tn) or ("location" in tn) or ("ctxdm" in tn):
        return ["location", "object"]
    # fallback
    return ["object"]


def build_condition_key(task_name: str, s1_row: pd.Series, s2_row: pd.Series, feature_type: list[str]) -> str:
    if ("object" in feature_type) and ("location" in feature_type):
        return f"{task_name}_loc{s1_row['location']}_obj{s1_row['object']}*loc{s2_row['location']}_obj{s2_row['object']}"
    elif ("object" in feature_type):
        return f"{task_name}_obj_{s1_row['object']}*obj_{s2_row['object']}"
    else:
        return f"{task_name}_trialpair"


# -----------------------
# Core grouping logic per run
# -----------------------

def group_encoding_betas_within_trial(beta_file: Path, cond_csv: Path, task_name: str, correct_only: bool = False):
    # Load betas (P x K_trials)
    with h5py.File(beta_file, 'r') as f:
        if 'betas' not in f:
            raise ValueError(f"No 'betas' in {beta_file}")
        betas = f['betas'][()]

    # Load design (one row per trial regressor IN ORDER)
    df = pd.read_csv(cond_csv)
    print(f"shape of df: {df.shape} (rows x columns)")
    
    df.columns = df.columns.str.strip()
    # Basic sanity
    required_cols = {"trialNumber", "stim_order", "type"}
    missing = required_cols - set(df.columns)
    if missing:
    
        raise ValueError(f"Design CSV missing columns {missing} in {cond_csv}")

    
    # if correct_only and 'is_correct' in df.columns:
    
    #     df = df[df['is_correct'] != False].copy()

    
    # Ensure column index matches betas' columns
    df = df.reset_index(drop=True)
    df['col_index'] = np.arange(len(df), dtype=int)

    
    if betas.shape[1] != len(df):
        warnings.warn(
            f"Betas columns ({betas.shape[1]}) != design rows ({len(df)}). Proceeding but mapping may be off.")

    # Keep only encoding rows
    df_enc = df[df['type'].astype(str).str.lower() == 'encoding'].copy()

    # Group within trial; pick consecutive stimulus pairs by task
    pair_indices_list = determine_task_type(task_name)
    feature_type = determine_feature_type(task_name)

    condition_dict: dict[str, list[np.ndarray]] = defaultdict(list)

    g = df_enc.groupby('trialNumber')
    for trial, trial_df in g:
        for i1, i2 in pair_indices_list:
            r1 = trial_df[trial_df['stim_order'] == i1]
            r2 = trial_df[trial_df['stim_order'] == i2]
            if r1.empty or r2.empty:
                continue
            s1 = r1.iloc[0]
            s2 = r2.iloc[0]
            key = build_condition_key(task_name, s1, s2, feature_type)

            # Select which beta to store for the pair
            if PAIR_SELECTION == 'mean':
                b = 0.5 * (betas[:, int(s1['col_index'])] + betas[:, int(s2['col_index'])])
            else:  # 'first'
                b = betas[:, int(s1['col_index'])]
            condition_dict[key].append(b)

    return condition_dict


# -----------------------
# Main
# -----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Group trial-level encoding betas into task-defined conditions")
    ap.add_argument("--subj", required=True, help="Subject ID, e.g., sub-01")
    ap.add_argument("--glm_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas",
                    help="Root of trial-level GLM outputs (contains <subj>/ses-*/func/*_betas.h5)")
    ap.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas",
                    help="Root for outputs (grouped pickle + histogram)")
    ap.add_argument("--tasks", nargs="*", default=None, help="Restrict to these tasks (e.g., ctxdm interdms nback)")
    ap.add_argument("--correct_only", action="store_true", help="Use only correct trials if column present")
    args = ap.parse_args()

    subj = args.subj
    glm_root = Path(args.glm_root) / subj

    if not glm_root.exists():
        raise SystemExit(f"GLM root not found: {glm_root}")

    all_condition_groups: dict[str, list[np.ndarray]] = defaultdict(list)

    # Iterate sessions/runs
    for session_dir in sorted(glm_root.glob("ses-*/")):
        func_dir = session_dir / "func"
        if not func_dir.exists():
            continue
        for beta_file in sorted(func_dir.glob("*_betas.h5")):
            subj_id, ses, task_name, run, base = parse_run_tokens(beta_file)
            if args.tasks and (task_name not in set(args.tasks)):
                continue
            cond_csv = beta_file.with_name(f"{base}_design.csv")
            if not cond_csv.exists():
                print(f"⚠️ Missing design CSV for {beta_file.name}; skipping run")
                continue

            try:
                condition_betas = group_encoding_betas_within_trial(beta_file, cond_csv, task_name, correct_only=args.correct_only)
                for k, v in condition_betas.items():
                    all_condition_groups[k].extend(v)
            except Exception as e:
                print(f"⚠️ Skipping {beta_file.name}: {e}")

    # Report
    print("\n✅ Condition counts:")
    counts = {k: len(v) for k, v in all_condition_groups.items()}
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v} repetitions")

    total_repetitions = sum(counts.values())
    print(f"\n🔢 Total conditions: {len(counts)}")
    print(f"🔁 Total repetitions across all conditions: {total_repetitions}")
    print(f"⚠️ Conditions with <5 repetitions: {sum(1 for v in counts.values() if v < 5)}")

    # Save grouped betas
    save_dir = Path(args.out_root) / "grouped_betas" / "task_relevant_only"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{subj}_task_condition_betas.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_condition_groups, f)
    print(f"\n✅ Grouped betas saved to: {save_path}")

    # Plot repetition distribution
    output_dir = Path(args.out_root) / "results" / "repetition_count" / "task_relevant_only" / "all_consecutive_stim_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    repetition_counts = list(counts.values()) or [0]
    plt.figure(figsize=(10, 6))
    bins = range(1, (max(repetition_counts) if repetition_counts else 1) + 2)
    plt.hist(repetition_counts, bins=bins, edgecolor='black')
    plt.title(f"{subj} | Distribution of Repetitions per Condition")
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Number of Conditions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / f"{subj}_repetition_distribution.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Repetition distribution figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
