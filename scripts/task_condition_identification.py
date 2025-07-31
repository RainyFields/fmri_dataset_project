import os
import h5py
import pandas as pd
import re
import pickle
from pathlib import Path
from collections import defaultdict, Counter

def extract_task_name(subject, session, run, design_tsv, task_name):
    df = pd.read_csv(design_tsv, sep="\t")
    df.columns = df.columns.str.strip()
    df['converted_file_name'] = df['converted_file_name'].str.strip()
    df['block_file_name'] = df['block_file_name'].str.strip()
    match_str = f"{subject}_{session}_task-"
    matched = df[df['converted_file_name'].str.contains(match_str) &
                 df['converted_file_name'].str.contains(f"run-{run}") &
                 df['converted_file_name'].str.contains(task_name)]
    if matched.empty:
        raise ValueError(f"No matching entry found for {subject} {session} run-{run}")
    block_file = matched.iloc[0]['block_file_name']

    return re.sub(r'_block_\d+$', '', block_file)

def determine_task_type(task_name):
    if task_name.startswith("interdms"):
        # return [(1, 2), (2, 3), (3, 4)]
        return [(1,2)]
    elif task_name.startswith("ctxdm"):
        # return [(1, 2), (2, 3)]
        return [(1,2)]
    elif task_name.startswith("nback"):
        return [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        # return [(1,2)]

def determine_feature_type(task_name):
    if "obj" in task_name or "ctg" in task_name:
        return [ "object"]
    elif "loc" or "ctxdm" in task_name:
        return ["location", "object"]
    else:
        raise ValueError(f"Unknown task type: {task_name}")

def group_encoding_betas_within_trial(beta_file, cond_csv, task_name):
    with h5py.File(beta_file, 'r') as f:
        betas = f['betas'][:]

    df = pd.read_csv(cond_csv)
    df.columns = df.columns.str.strip()
    df = df[df['regressor_type'] == "encoding"].copy()

    if task_name.startswith("interdms"):
        if len(df) != 64:
            raise ValueError(f"Expected 64 trials for interdms, found {len(df)} in {cond_csv.name}")
        
    elif task_name.startswith("ctxdm"):
        if len(df) != 60:
            raise ValueError(f"Expected 64 trials for ctxdm, found {len(df)} in {cond_csv.name}")
    elif task_name.startswith("nback"):
        if len(df) != 54:
            raise ValueError(f"Expected 54 trials for nback, found {len(df)} in {cond_csv.name}")
        

    if 2 * len(df) != betas.shape[1]:
        raise ValueError(f"Mismatch between beta rows ({betas.shape[1]}) and encoding trials (2 * {len(df)}) in {beta_file.name}")

    df["beta_index"] = range(len(df))  # map back to beta rows
    grouped = df.groupby("trialNumber")
    pair_indices_list = determine_task_type(task_name)
    feature_type = determine_feature_type(task_name)

    condition_dict = defaultdict(list)
    for trial, trial_df in grouped:
        for i1, i2 in pair_indices_list:
            stim1 = trial_df[trial_df["stim_order"] == i1]
            stim2 = trial_df[trial_df["stim_order"] == i2]
            if stim1.empty or stim2.empty:
                continue
            s1 = stim1.iloc[0]
            s2 = stim2.iloc[0]

            if "object" in feature_type and "location" in feature_type:
                key = f"{task_name}_loc{s1['location']}_obj{s1['object']}*loc{s2['location']}_obj{s2['object']}"
            elif "object" in feature_type:
                key = f"{task_name}_obj_{s1['object']}*obj_{s2['object']}"
            else:
                raise ValueError("Unsupported feature_type")

            # Add beta (or concatenate both if needed)
            beta = betas[:, s1["beta_index"]]
            condition_dict[key].append(beta)

    return condition_dict

def main():
    subject = "sub-01"
    beta_root = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/betas/{subject}")
    design_root = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs")
    design_tsv = design_root / f"{subject}_design_design_with_converted.tsv"

    all_condition_groups = defaultdict(list)

    for session_dir in beta_root.glob("ses-*"):
        func_dir = session_dir / "func"
        if not func_dir.exists():
            continue

        for beta_file in func_dir.glob("*_task-*_*_betas.h5"):
            file_stem = beta_file.stem.replace("_betas", "")
        
            tokens = file_stem.split("_")
        
            ses = tokens[1]
            run = tokens[-1].replace("run-", "")
            task_name = tokens[-2][5:]
        
            cond_csv = beta_file.with_name(f"{file_stem}_design.csv")

            try:
                task_name = extract_task_name(subject, ses, run, design_tsv, task_name)
                condition_betas = group_encoding_betas_within_trial(beta_file, cond_csv, task_name)
                for k, v in condition_betas.items():
                    all_condition_groups[k].extend(v)
            except Exception as e:
                print(f"⚠️ Skipping {beta_file.name}: {e}")

        # Print counts
    print("\n✅ Condition counts:")
    counts = {k: len(v) for k, v in all_condition_groups.items()}
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v} repetitions")

    total_repetitions = sum(counts.values())
    print(f"\n🔢 Total conditions: {len(counts)}")
    print(f"🔁 Total repetitions across all conditions: {total_repetitions}")
    print(f"⚠️ Conditions with <5 repetitions: {sum(1 for v in counts.values() if v < 5)}")

    # ✅ Save grouped betas
 

    save_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/betas/grouped_betas/task_relevant_only/")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{subject}_task_condition_betas.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_condition_groups, f)

    print(f"\n✅ Grouped betas saved to: {save_path}")

    # ✅ Plot and save repetition distribution
    import matplotlib.pyplot as plt

    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/repetition_count/task_relevant_only/all_consecutive_stim_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    repetition_counts = list(counts.values())

    plt.figure(figsize=(10, 6))
    plt.hist(repetition_counts, bins=range(1, max(repetition_counts)+2), edgecolor='black')
    plt.title(f"{subject} | Distribution of Repetitions per Condition")
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Number of Conditions")
    plt.grid(True)
    plt.tight_layout()

    fig_path = output_dir / f"{subject}_repetition_distribution.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"📊 Repetition distribution figure saved to: {fig_path}")



if __name__ == "__main__":
    main()
