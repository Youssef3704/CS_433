# ML Project 1 — Predicting coronary heart disease (MICHD) from BRFSS survey answers

EPFL CS-433, Fall 2026. Binary classification of 328,135 BRFSS 2015 respondents
(8.8% positive), scored by F1 on 109,379 test rows. Only NumPy is used for
modelling; the full method and its ablation are described in `report.pdf`.

## Reproducing the submission

All code lives in `ML_Project1_EPFL/`.

1. Put `x_train.csv`, `y_train.csv` and `x_test.csv` (unchanged names) in
   `ML_Project1_EPFL/data/`, or pass another folder with `--data-path`.
2. Install NumPy and run from `ML_Project1_EPFL/`:

```bash
cd ML_Project1_EPFL
python -m pip install -r requirements.txt
python run.py                 # writes submission.csv
python ablation.py            # optional: reproduces the report's ablation table
```

The first load parses the CSVs (about 45 s) and caches them as `data/cache.npz`;
later runs start in seconds. `run.py` takes about 7 minutes on a laptop CPU and
is fully deterministic (fixed fold seed, zero initial weights, full-batch
gradient descent), so it rewrites the same `submission.csv` byte for byte.

## What `run.py` does

1. **Preprocessing** (`preprocessing.py`), always fitted on training rows only:
   1. Recode BRFSS "don't know" / "refused" codes to missing. The codes differ per
      variable (7/9 for `GENHLTH`, 77/99 for day counts, 7777/9999 for `WEIGHT2`,
      14 for `_AGEG5YR`, ...) and are read from the official CDC 2015 codebook,
      transcribed in `brfss_codebook.py`.
   2. Drop columns more than 50% missing, constant columns, and 28 survey
      administration columns (record ids, sampling weights, interview dates,
      telephone screening, `_STATE`).
   3. Standardize each kept column with the mean and std of its observed values,
      then set missing entries to 0, i.e. to the column mean.
   4. Add two standardized per-row counts: answers refused and questions never
      asked. Add an intercept column. Result: 155 columns.
2. **Model**: logistic regression (`reg_logistic_regression`) by full-batch
   gradient descent, `gamma = 1.0`, `max_iters = 500`, zero initial weights.
3. **Model selection**: 5-fold CV (seed 1) over `lambda in {0, 0.001, 0.01}`
   and 56 decision thresholds 0.05, 0.06, ..., 0.60. Preprocessing is refitted
   inside every training fold, so no validation row leaks into it. The
   (lambda, threshold) with the best mean validation F1 is kept: lambda = 0.001,
   threshold 0.20, CV F1 0.4252 (lambda = 0 scores 0.4248, a difference well
   inside the fold standard deviation of about 0.004).
4. **Final fit**: preprocessing and weights refitted on all training rows; test
   rows with predicted probability at or above the threshold are labelled 1,
   the rest -1.

## Files (in `ML_Project1_EPFL/`)

| File | Contents |
| --- | --- |
| `implementations.py` | The six required methods and their loss/gradient helpers |
| `preprocessing.py` | Sentinel recoding, column filtering, standardization, counts |
| `brfss_codebook.py` | Per-variable sentinel codes from the CDC BRFSS 2015 SAS formats |
| `validation.py` | Train/validation split, k-fold CV, accuracy and F1 |
| `helpers.py` | CSV loading (with `.npz` cache), submission writer, prediction |
| `run.py` | Final pipeline: CV model selection, refit, `submission.csv` |
| `ablation.py` | Full-data ablation of every pipeline change (report Table 1) |
| `trials.txt` | Log of every experiment, including the ones that failed |
| `data-testing.ipynb` | Exploratory data analysis |
| `report.tex`, `report.pdf` | Two-page report |

## Method contracts

All six methods in `implementations.py` return `(w, loss)` with 1-D arrays. The
MSE has the factor 1/2, SGD samples one row per step, least squares solves the
normal equations without `numpy.linalg.lstsq`, logistic methods take labels in
{0, 1}, and the returned losses of the regularized methods exclude the penalty.
