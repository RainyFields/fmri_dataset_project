import os
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import signal
from sklearn.linear_model import LinearRegression

import nibabel as nib
# nipy installation not successful, use updated nilearn instead
# from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from nilearn.glm.first_level import spm_hrf

from utils import glm_confounds_construction, standardize_run_label

tr_length = 1.49
subj = "sub-05"
correct_trials_only = False # also to change output_dir
fmri_root_dir = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled/{subj}")
confounds_root_dir = Path(f"/project/def-pbellec/xuan/cneuromod.multfs.fmriprep/{subj}")
events_root_dir = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior/{subj}")
if correct_trials_only:
    output_dir = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/correct_betas/{subj}")
else:
    output_dir = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/data/betas/{subj}")
# all_task_types = ["1back", "ctxdm", "dms", "interdms"] # does not work for dms for now
all_task_types = ["1back", "ctxdm", "interdms"]
tmask = 1  # number of frames to skip at the beginning

pattern = re.compile(r".*\d+$")

all_sessions = [p.name for p in fmri_root_dir.iterdir()if p.is_dir() and pattern.match(p.name)]

print(all_sessions)
# extract all possible session

for task_name in all_task_types:
    for ses in all_sessions:
        # extract all the runs within the session based on fmri timeseries data
        pattern = f"{subj}_{ses}_task-{task_name}_*_space-Glasser64k_bold.dtseries.nii"

        # Find all matching files
        matching_files = list(Path(os.path.join(fmri_root_dir, ses)).glob(pattern))
        print(matching_files)

        # Extract run identifiers using regex
        run_pattern = re.compile(rf"{subj}_{ses}_task-{task_name}_(.+?)_space-Glasser64k_bold\.dtseries\.nii")
        runs = []
        for file in matching_files:
            match = run_pattern.match(file.name)
            if match:
                runs.append(match.group(1))
        print(f"all possible runs: {runs}")

        for run in runs:

            # Determine the expected output HDF5 path
            behavioral_file = os.path.join(events_root_dir, ses, "func", f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv")
            behavioral_path = Path(behavioral_file)
            rel_behavioral_path = behavioral_path.relative_to(events_root_dir)
            target_subdir = output_dir / rel_behavioral_path.parent
            base_name = behavioral_path.stem.replace("_events", "")  # e.g. sub-03_ses-003_task-1back_run-01
            h5_file = target_subdir / f"{base_name}_betas.h5"

            # Skip if output already exists
            if h5_file.exists():
                print(f"[⏩] Skipping {h5_file} — already exists.")
                continue

            timeseries_file = os.path.join(fmri_root_dir, ses, f"{subj}_{ses}_task-{task_name}_{run}_space-Glasser64k_bold.dtseries.nii")
            timeseries_data = nib.load(timeseries_file).get_fdata()
            assert (np.sum(timeseries_data==np.nan) == 0)

            confounds_file = os.path.join(confounds_root_dir, ses, "func", f"{subj}_{ses}_task-{task_name}_{run}_desc-confounds_timeseries.tsv")
            df_confounds = pd.read_csv(confounds_file, sep='\t')

            behavioral_file = os.path.join(events_root_dir, ses, "func", f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv")
            if not Path(behavioral_file).exists():
                print(f"⚠️ Behavioral file {behavioral_file} does not exist, skipping.")
                continue
            df_events = pd.read_csv(behavioral_file, sep='\t')
            df_events.columns = df_events.columns.str.strip()
            df_events = df_events.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)


            timeseries_confounds = glm_confounds_construction(df_confounds)
            num_trs = timeseries_data.shape[0]
            print(f"shape of timeseries data: {timeseries_data.shape}")
            print(f"number of trs: {num_trs}")

            df_conditions = {}
            regressors = [] # do I need it here?

            df_conditions['trialNumber'] = []
            df_conditions['stim_order'] = []
            df_conditions['location'] = []
            df_conditions['category'] = []
            df_conditions["object"] = []
            df_conditions["is_correct"] = []

            df_conditions["regressor_type"] = []

            for idx, row in df_events.iterrows():
                if correct_trials_only and row['is_correct'] is False:
                    continue
                df_conditions['trialNumber'].append(row['trialNumber'])
                df_conditions['stim_order'].append(row['stim_order'])
                df_conditions['location'].append(row['locmod'])
                df_conditions['category'].append(row['ctgmod'])
                df_conditions['object'].append(row['objmod'])
                df_conditions['is_correct'].append(row['is_correct'])
                df_conditions['regressor_type'].append(row['type'])
                reg_array = np.zeros((num_trs,))  
                onset_tr = int(np.ceil(row["onset_time"]/tr_length))  
                offset_tr = int(np.floor(row["offset_time"]/tr_length))
                reg_array[onset_tr:offset_tr] = 1
                regressors.append(reg_array)

            regressors = np.asarray(regressors).T # time x regressors
            df_conditions = pd.DataFrame(df_conditions)

            # now convolve with canonical HRF
            regressors_hrf = np.zeros(regressors.shape)
            spm_hrfTS = spm_hrf(tr_length,oversampling=1)

            for stim in range(regressors.shape[1]):
                # perform convolution
                tmpconvolve = np.convolve(regressors[:,stim],spm_hrfTS)
                regressors_hrf[:,stim] = tmpconvolve[:num_trs]

            tmask_arr = np.ones((num_trs,))
            tmask_arr[:tmask] = 0
            tmask_arr = np.asarray(tmask_arr,dtype=bool)

            # skip frames
            timeseries_data = timeseries_data[tmask_arr,:]
            timeseries_confounds = timeseries_confounds[tmask_arr,:]
            regressors_hrf = regressors_hrf[tmask_arr,:]


            #### START GLM
            # Demean each run
            assert(np.sum(timeseries_data==np.nan)==0)
            timeseries_data = signal.detrend(timeseries_data,axis=0,type='constant')
            # Detrend each run
            timeseries_data = signal.detrend(timeseries_data,axis=0,type='linear')

            n_spatial_dim = timeseries_data.shape[1]

            # timeseries_confounds have nan values for derivatives
            timeseries_confounds = np.nan_to_num(timeseries_confounds, nan=0)
            
            # last len(condition_labels) set of regressors correspond to the task
            all_regressors = np.hstack((timeseries_confounds,regressors_hrf))


            reg = LinearRegression().fit(all_regressors,timeseries_data)
            betas = reg.coef_
            assert(betas.shape[0]==n_spatial_dim) # make sure first value is number of regions/voxels
            betas = reg.coef_[:,-len(df_conditions):]
            y_pred = reg.predict(all_regressors)
            resid = timeseries_data - y_pred

            residual_ts = resid.T


            # Recreate sub/session/func structure
            behavioral_path = Path(behavioral_file)
            rel_behavioral_path = behavioral_path.relative_to(events_root_dir)
            target_subdir = output_dir / rel_behavioral_path.parent
            target_subdir.mkdir(parents=True, exist_ok=True)

            # Use original behavioral filename (without "_events.tsv") as base
            base_name = behavioral_path.stem.replace("_events", "")  # e.g. sub-03_ses-003_task-1back_run-01

            # Save CSV (condition design)
            df_conditions.to_csv(target_subdir / f"{base_name}_design.csv", index=False)

            # Save HDF5 file with GLM results
            h5_file = target_subdir / f"{base_name}_betas.h5"
            with h5py.File(h5_file, 'a') as h5f:
                for key, data in {
                    "betas": betas,
                    "regressors": regressors,
                    "all_regressors": all_regressors,
                }.items():
                    if key in h5f:
                        del h5f[key]
                    h5f.create_dataset(key, data=data)
            print(f"[🎯]Saved betas and design to {h5_file} and {target_subdir / f'{base_name}_design.csv'}")


# for each session, data I need: 
# glasser data[timeseries_data]: sub-03_ses-001_task-1back_run-1_space-Glasser64k_bold.dtseries.nii
# nuisance regressors [df_confounds]: /project/def-pbellec/xuan/cneuromod.multfs.fmriprep/sub-01/ses-001/func/sub-01_ses-001_task-1back_run-1_desc-confounds_timeseries.tsv
# events file [df_events]: /project/def-pbellec/xuan/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw/sub-03/ses-001/func/sub-03_ses-001_task-1back_run-01_events.tsv







