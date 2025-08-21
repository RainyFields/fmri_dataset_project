import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path

import nibabel as nib

# ==== Config ====
subject = "sub-01"
region_level = True
atlas_path = "/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii"

# ==== Helper ====
def rowwise_correlation(X, Y):
    """Compute Pearson r for each column (voxel or region) between X and Y using scipy."""
    assert X.shape == Y.shape
    r = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_col = X[:, i]
        y_col = Y[:, i]
        if np.all(x_col == x_col[0]) or np.all(y_col == y_col[0]):
            r[i] = np.nan
            continue
        r[i], _ = pearsonr(x_col, y_col)
    return r

# ==== Load atlas and get label mapping ====
if region_level:
    dlabel = nib.load(atlas_path)
    atlas_data = dlabel.get_fdata().squeeze()
    unique_labels = np.unique(atlas_data)
    region_labels = unique_labels[unique_labels > 0]  # remove unlabeled 0
    region_labels = region_labels.astype(int)
    n_regions = len(region_labels)

# ==== Load Betas ====
input_path = f"/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/grouped_betas/task_relevant_only/{subject}_task_condition_betas.pkl"
with open(input_path, "rb") as f:
    condition_betas = pickle.load(f)

valid_conditions = [k for k, v in condition_betas.items() if len(v) >= 4]
if not valid_conditions:
    raise ValueError("No valid conditions with at least 4 repetitions.")

# ==== Bootstrap NC Calculation ====
n_bootstrap = 10
all_bootstrapped_ncs = []

for b in range(n_bootstrap):
    half1_matrix = []
    half2_matrix = []

    for cond in valid_conditions:
        betas = np.stack(condition_betas[cond], axis=0)  # (n_reps, n_voxels)
        n_trials, n_vox = betas.shape

        if region_level:
            # Average across voxels in each region per trial
            betas_region = np.full((n_trials, n_regions), np.nan)
            for i, region_id in enumerate(region_labels):
                region_mask = (atlas_data == region_id)
                if region_mask.sum() == 0:
                    continue
                betas_region[:, i] = np.nanmean(betas[:, region_mask], axis=1)
            betas = betas_region  # (n_trials, n_regions)

        if betas.shape[0] < 4:
            continue

        # Random split
        perm = np.random.permutation(betas.shape[0])
        half1_mean = betas[perm[:len(perm) // 2]].mean(axis=0)
        half2_mean = betas[perm[len(perm) // 2:]].mean(axis=0)

        half1_matrix.append(half1_mean)
        half2_matrix.append(half2_mean)

    if not half1_matrix or not half2_matrix:
        print(f"⚠️ Bootstrap {b} skipped due to insufficient data.")
        continue

    half1_matrix = np.stack(half1_matrix, axis=0)
    half2_matrix = np.stack(half2_matrix, axis=0)

    nc = rowwise_correlation(half1_matrix, half2_matrix)
    all_bootstrapped_ncs.append(nc)

# ==== Final Noise Ceiling ====
noise_ceiling = np.nanmean(np.stack(all_bootstrapped_ncs, axis=0), axis=0)

# ==== Save and Plot ====
unit = "region" if region_level else "voxel"
clipped_nc = np.clip(noise_ceiling, -10, 10)

plt.figure(figsize=(10, 6))
plt.hist(clipped_nc, bins=50, edgecolor='black')
plt.title(f"Distribution of Noise Ceilings ({subject}, {unit}-wise)")
plt.xlabel("Noise Ceiling (Pearson r)")
plt.ylabel(f"Number of {unit}s")
plt.grid(True)
plt.tight_layout()

output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/traditional_way")
output_dir.mkdir(parents=True, exist_ok=True)
fig_path = output_dir / f"{subject}_noise_ceiling_distribution_per_{unit}.png"
plt.savefig(fig_path)
plt.close()

print(f"📊 Noise ceiling distribution plot saved to: {fig_path}")

# Save data
nc_save_path = output_dir / f"{subject}_noise_ceiling_per_{unit}.pkl"
with open(nc_save_path, "wb") as f:
    pickle.dump(noise_ceiling, f)

print(f"💾 Noise ceiling saved to: {nc_save_path}")
