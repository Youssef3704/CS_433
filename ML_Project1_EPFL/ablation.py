"""Reproduce the ablation table of the report on the full training set.

Each row adds one change to the row above it and is scored with the same 5-fold
CV as run.py (seed 1, preprocessing fitted inside each training fold). Prints the
table and writes the per-threshold F1 curves to ablation_results.json.

    python ablation.py [--data-path data]      # about 15 minutes
    python ablation.py --plot                  # report figure from the saved results
"""
import argparse
import json
import time

import numpy as np

from helpers import load_csv_data, load_feature_names
from implementations import least_squares
from preprocessing import fit_preprocessor, transform_features, encode_labels
from run import THRESHOLDS, cross_validated_probabilities, f1_grid, fit_fold
from validation import compute_f1

RAW = dict(recode_sentinels=False, drop_metadata=False, add_counts=False)
NO_SENTINELS = dict(recode_sentinels=False, drop_metadata=True, add_counts=False)
NO_COUNTS = dict(recode_sentinels=True, drop_metadata=True, add_counts=False)
FINAL = dict(recode_sentinels=True, drop_metadata=True, add_counts=True)

# (label, preprocessing options, gamma, lambda, tune threshold)
CONFIGS = (
    ("Logistic, threshold 0.5, gamma=0.1, lambda=1", RAW, 0.1, 1.0, False),
    ("+ tuned threshold", RAW, 0.1, 1.0, True),
    ("+ lambda=0", RAW, 0.1, 0.0, True),
    ("+ gamma=1", RAW, 1.0, 0.0, True),
    ("+ drop survey metadata", NO_SENTINELS, 1.0, 0.0, True),
    ("+ codebook sentinels -> missing", NO_COUNTS, 1.0, 0.0, True),
    ("+ refused / never-asked counts (final)", FINAL, 1.0, 0.0, True),
)


def fit_least_squares(x, y, feature_names):
    """Least-squares baseline on the final features, returned in fit_fold's format."""
    state = fit_preprocessor(x, feature_names, **FINAL)
    w, _ = least_squares(y, transform_features(x, state))
    return {0.0: w}, state


def summarize(label, fold_probabilities, lambda_, tune):
    """Return one table row: best mean CV F1, its fold std and threshold, mean output."""
    thresholds = THRESHOLDS if tune else np.array([0.5])
    f1 = f1_grid(fold_probabilities, lambda_, thresholds)
    j = int(np.argmax(f1.mean(axis=0)))
    mean_p = float(np.mean(np.concatenate([p[lambda_] for _, p in fold_probabilities])))
    return {"label": label, "f1": float(f1.mean(axis=0)[j]), "std": float(f1[:, j].std()),
            "threshold": float(thresholds[j]), "mean_p": mean_p,
            "curve": f1.mean(axis=0).tolist()}


def main():
    """Score every configuration and print the ablation table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--plot", action="store_true",
                        help="only draw threshold_f1.pdf from ablation_results.json")
    args = parser.parse_args()
    if args.plot:
        plot()
        return

    x, _, y, _, _ = load_csv_data(args.data_path)
    names = load_feature_names(args.data_path)
    y = encode_labels(y)

    rows = [{"label": "All-positive baseline", "f1": compute_f1(y, np.ones_like(y)),
             "std": 0.0, "threshold": None, "mean_p": 1.0, "curve": None}]

    start = time.time()
    # a regression output is not a probability, so it is thresholded as it is
    folds = cross_validated_probabilities(x, y, lambda a, b: fit_least_squares(a, b, names),
                                          output_fn=lambda score: score)
    rows.append(summarize("Least squares (final features)", folds, 0.0, True))
    print(f"{rows[-1]['label']}: {time.time() - start:.0f}s", flush=True)

    for label, options, gamma, lambda_, tune in CONFIGS:
        start = time.time()
        folds = cross_validated_probabilities(
            x, y, lambda a, b: fit_fold(a, b, names, lambdas=(lambda_,), gamma=gamma, **options))
        rows.append(summarize(label, folds, lambda_, tune))
        print(f"{label}: {time.time() - start:.0f}s", flush=True)

    print(f"\n{'configuration':<42} {'CV F1':>7} {'std':>7} {'thr':>5} {'mean p':>7}")
    for r in rows:
        thr = "-" if r["threshold"] is None else f"{r['threshold']:.2f}"
        print(f"{r['label']:<42} {r['f1']:7.4f} {r['std']:7.4f} {thr:>5} {r['mean_p']:7.4f}")

    with open("ablation_results.json", "w") as f:
        json.dump({"thresholds": THRESHOLDS.tolist(), "rows": rows}, f, indent=1)


def plot(path="ablation_results.json", output="threshold_f1.pdf"):
    """Draw the report figure, CV F1 against decision threshold, from saved results."""
    import matplotlib.pyplot as plt  # visualization only

    with open(path) as f:
        results = json.load(f)
    shown = {"+ gamma=1": "logistic, raw features",
             "+ refused / never-asked counts (final)": "logistic, final features",
             "Least squares (final features)": "least squares, final features"}
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for row in results["rows"]:
        if row["label"] in shown:
            ax.plot(results["thresholds"], row["curve"], label=shown[row["label"]])
    base = results["rows"][0]["f1"]
    ax.axhline(base, color="grey", linestyle=":", linewidth=1, label="predict all positive")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("5-fold CV F1")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(output)


if __name__ == "__main__":
    main()
