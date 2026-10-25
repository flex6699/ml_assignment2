# ML Assignment 2 – Bagging, Boosting and Bias–Variance on Face Recognition

Name: Aniruddha Bohra  
Roll number: P25CS0201 (seed for `numpy.random.default_rng` = 250201, set in `common.py`)

## Environment

The code was run on Linux with **Python 3.11** in a Miniconda environment named `ml_as2`.
Package versions are pinned in `requirements.txt`.

### Step 1: install Miniconda

Skip this step if `conda --version` already works.

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc

The prompt now starts with `(base)`. Check with `conda --version`.

### Step 2: create the environment and install the packages

Run these from the repository folder.

    conda create -n ml_as2 python=3.11 -c conda-forge --override-channels -y
    conda activate ml_as2
    pip install -r requirements.txt

`-c conda-forge --override-channels` installs Python from the free conda-forge channel.
Without it, recent conda versions stop with `CondaToSNonInteractiveError` until the Anaconda
Terms of Service are accepted (`conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
and the same for `https://repo.anaconda.com/pkgs/r`).

### Step 3: check the environment

    python --version
    python -c "import numpy, sklearn, matplotlib; print(numpy.__version__, sklearn.__version__, matplotlib.__version__)"

This should print `Python 3.11.x` and `2.4.4 1.8.0 3.10.8`.
Run every command below inside the activated `ml_as2` environment.

## Dataset

The Olivetti faces dataset is downloaded by scikit-learn on first use and cached in `~/scikit_learn_data/`:

    python -c "from sklearn.datasets import fetch_olivetti_faces; fetch_olivetti_faces(); print('ok')"

If the machine has no internet access, run this command on another machine and copy
`~/scikit_learn_data/olivetti_py3.pkz` to the same folder on this one.

## Saved split

`split/split.json` holds the 6 train / 2 validation / 2 test images per identity, made with seed 250201.
All scripts load this file. Each time it is loaded it is checked against the seed, and the script
stops if it does not match.

## Reproduction commands

    mkdir -p results
    python q1_tree_bagging.py | tee results/q1_log.txt
    python q2_boosting.py | tee results/q2_log.txt
    python q3_bias_variance.py | tee results/q3_log.txt
    python extra_figures.py | tee results/extra_log.txt

Run them in this order: Q2 loads the frozen tree and bagging settings that Q1 writes to `configs/`,
and Q3 loads all frozen settings plus `results/q2.json`. Q3 refits 330 models (30 resampled
training sets × 11 configurations) in parallel over all CPU cores (`Q3_JOBS=4` limits the number of
workers) and caches each finished configuration in `results/q3_cache/`, so an interrupted run resumes.
`common.py` holds the shared code: data loading, saved split, masking, bagging (`BaggedTrees`)
and SAMME boosting (`SAMME`), both implemented from scratch on top of `DecisionTreeClassifier`.


## Outputs

| Path | Contents |
|---|---|
| `split/split.json` | saved train / validation / test split |
| `configs/frozen_tree.json` | selected single tree setting (Q1a) |
| `configs/frozen_bagging.json` | selected bagging ensemble size (Q1c) |
| `results/q1.json`, `results/q1_log.txt` | all Q1 numbers and the run log |
| `figures/q1a_depth_sweep.png` | accuracy and leaf statistics against depth |
| `figures/q1c_oob_disagreement_margin.png` | OOB accuracy, disagreement and vote margin against B |
| `figures/q1d_cases.png` | faces corrected by bagging and faces both models get wrong |
| `configs/frozen_boosting.json` | selected boosting depth and number of rounds (Q2b) |
| `results/q2.json`, `results/q2_log.txt` | all Q2 numbers and the run log |
| `results/q2a_toy_round.txt` | one complete SAMME round on the 3-class toy example (Q2a) |
| `figures/q2b_boosting_dynamics.png` | weighted error, learner weight, max weight, top-10 weight over rounds |
| `figures/q2c_heavy_face.png` | prediction history of the training face with the largest boosting weight |
| `figures/q2d_mask_response.png` | face where bagging and boosting respond differently to the mask |
| `results/q3_resamples.json` | the 30 within-identity resampled training sets (Q3) |
| `results/q3.json`, `results/q3_log.txt` | all Q3 bias, variance, accuracy, leaf and disagreement numbers |
| `results/q3_final_table.tex` | LaTeX version of the final Q3(d) table |
| `figures/q3b_depth_bias_variance.png` | bias², variance and accuracy of trees and bagging at each depth |
| `figures/q3d_summary.png` | bias² + variance and clean / masked accuracy of the three final models |
| `figures/extra_split_locations.png` | pixels tested by split nodes for tree, bagging and boosting, with the mask outlined |
| `figures/extra_collapse_margins.png` | prediction collapse under the mask and vote-margin distributions |
| `results/extra_numbers.tex` | median vote margins used in the report text |
