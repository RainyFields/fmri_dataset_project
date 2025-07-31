# fMRI Project Overview

This project is organized into three main sections:

- **Data**
- **Scripts**
- **Results**

---

## 📂 Data Access

### Raw fMRI and Behavioral Data

- Preprocessed fMRI data:  
  `/project/def-pbellec/xuan/cneuromod.multfs.fmriprep`

- Raw behavioral data (nested inside the above):  
  `/project/def-pbellec/xuan/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw`

### Syncing with Datalad

To access the raw dataset using Datalad, follow these steps:

```bash
module load connectomeworkbench  # for wb_command
source /project/def-pbellec/shared/venvs/datalad/bin/activate  # ensures correct git-annex version

datalad install --reckless ephemeral -s ria+file:///project/rrg-pbellec/ria-rorqual#~cneuromod.multfs.fmriprep@dev
cd cneuromod.multfs.fmriprep
datalad get -n --reckless ephemeral sourcedata/cneuromod.multfs.raw/  # download raw data
````

> ⚠️ **Note**: The existing dataset naming convention includes only the task type.
> For a mapping to full/descriptive task names, refer to:
> `/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs`

---

## ⚙️ Scripts & Processing Steps

### Environment Setup

Before running any scripts, make sure to:

* Request resources on the Rorqual cluster
* Activate the appropriate Python environment using:

```bash
source py_venv_activate.sh
```

---

### Resampling to Glasser Space

To resample the fMRI data into Glasser space (CIFTI format), run:

```bash
bash cifti_resample.sh
```

This step ensures alignment with the standard cortical parcellation used in subsequent analyses.

---

### Accuracy Evaluation

To inspect decoding accuracy:

* Run: `accuracy_eval.py`
* Results will be saved to:

```text
/project/def-pbellec/xuan/fmri_dataset_project/results/1back_accuracy_results.csv
```

> ✅ You can change the task name in the script to compute accuracy for other tasks.

---

### GLM Analysis

For voxel-wise or region-wise GLM modeling, use:

* Script: `glm_analysis.py`
* Optional argument: `correct_trials_only=True` (filters only correct-response trials)

> ⚠️ Make sure to update the output path accordingly to keep results organized and consistent.

---

## 📊 Noise Ceiling Estimation

Two methods are implemented:

* `noise_ceiling.py`:
  Implements the approach used in the *CNeuroMod Things* paper

* `noise_ceiling_traditional.py`:
  Uses the classical split-half correlation method (randomly splits repetitions in half)

---

## 🔍 Task Condition Identification

To inspect how often each task condition appears:

* Use: `task_condition_identification.py`

You can customize the following functions:

* `determine_task_type`:
  Defines which frame pairs are counted per task

* `determine_feature_type`:
  Specifies which feature dimensions to consider for each task

---

## 📁 Output Summary

| Component          | Output Path Example                                                                 |
| ------------------ | ----------------------------------------------------------------------------------- |
| Accuracy Results   | `/project/def-pbellec/xuan/fmri_dataset_project/results/1back_accuracy_results.csv` |
| Study Design Info  | `/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs`                 |
| Noise Ceiling Data | Depends on which script and task type is used (modify output path accordingly)      |

---

## ✅ Tips

* Always verify task identity mappings using the study design files.
* Track task-specific logic via `task_condition_identification.py` if you change tasks or conditions.
* Keep output folders consistent with parameter settings to avoid overwriting results.
