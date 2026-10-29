"""Reproduce the final submission: cross-validate, refit on all training rows, predict.

Usage (from this folder, with x_train.csv, y_train.csv and x_test.csv in data/):

    python run.py [--data-path data] [--output submission.csv]

Every step is deterministic (fixed fold seed, zero initial weights, full-batch
gradient descent), so the same data always yields the same submission file.
"""
import argparse
import time

import numpy as np

from helpers import (load_csv_data, load_feature_names,
                     predict_labels_logistic, create_csv_submission)
from implementations import reg_logistic_regression, sigmoid
from preprocessing import fit_preprocessor, transform_features, encode_labels
from validation import compute_f1, cross_validate

GAMMA = 1.0         # largest stable step: 5.0 already diverges
MAX_ITERS = 500
# Each step multiplies w by (1 - 2 * GAMMA * lambda), and GAMMA = 1 already sits near
# the stability limit of the loss itself: lambda = 0.1 oscillates, and at lambda = 1
# the update is w <- -w - gradient. Larger lambdas need a smaller step (trials.txt).
LAMBDAS = (0.0, 0.001, 0.01)
THRESHOLDS = np.round(np.arange(0.05, 0.605, 0.01), 2)
K_FOLD = 5
SEED = 1


def fit_fold(x, y, feature_names, lambdas=LAMBDAS, gamma=GAMMA, max_iters=MAX_ITERS,
             **preprocessing_options):
    """Fit preprocessing on one training fold, then one weight vector per lambda.

    The preprocessing does not depend on lambda, so it is fitted once and shared.
    Returns ({lambda: weights}, preprocessing_state).
    """
    state = fit_preprocessor(x, feature_names, **preprocessing_options)
    tx = transform_features(x, state)
    weights = {lambda_: reg_logistic_regression(y, tx, lambda_, np.zeros(tx.shape[1]),
                                                max_iters, gamma)[0]
               for lambda_ in lambdas}
    return weights, state


def cross_validated_probabilities(x, y, train_fn, output_fn=sigmoid):
    """Return one (y_valid, {lambda: probabilities}) pair per validation fold.

    The threshold plays no part in training, so the probabilities are stored once
    and re-thresholded as often as needed. output_fn maps the linear score to the
    thresholded quantity (the identity for a regression baseline).
    """
    folds = cross_validate(x, y, train_fn, k_fold=K_FOLD, seed=SEED)
    result = []
    for fold in folds.values():
        weights, state = fold["model"]
        tx_valid = transform_features(x[fold["valid_idx"]], state)
        result.append((y[fold["valid_idx"]],
                       {lambda_: output_fn(tx_valid @ w) for lambda_, w in weights.items()}))
    return result


def f1_grid(fold_probabilities, lambda_, thresholds=THRESHOLDS):
    """Return the per-fold F1 at every threshold, shape (K_FOLD, len(thresholds))."""
    return np.array([[compute_f1(y_valid, (probabilities[lambda_] >= t).astype(int))
                      for t in thresholds]
                     for y_valid, probabilities in fold_probabilities])


def main():
    """Load data, select lambda and threshold by CV, refit, and write the submission."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    x_train, x_test, y_train, _, test_ids = load_csv_data(args.data_path)
    # column names, so the preprocessor can find the codebook sentinels and the metadata
    feature_names = load_feature_names(args.data_path)
    # the logistic gradient expects labels in {0, 1}, not the {-1, 1} of the competition
    y_train = encode_labels(y_train)
    print(f"Loaded {x_train.shape[0]} training rows, {x_train.shape[1]} raw features")

    start = time.time()
    fold_probabilities = cross_validated_probabilities(
        x_train, y_train, lambda x, y: fit_fold(x, y, feature_names))
    print(f"{K_FOLD}-fold CV, {len(LAMBDAS)} lambdas fitted in {time.time() - start:.0f}s")

    best = (-1.0, None, None)
    for lambda_ in LAMBDAS:
        f1 = f1_grid(fold_probabilities, lambda_)
        j = int(np.argmax(f1.mean(axis=0)))
        mean_p = np.mean(np.concatenate([p[lambda_] for _, p in fold_probabilities]))
        print(f"lambda={lambda_:<5} CV F1 {f1.mean(axis=0)[j]:.4f} +- {f1[:, j].std():.4f} "
              f"@ threshold {THRESHOLDS[j]:.2f}   mean predicted p {mean_p:.4f}")
        if f1.mean(axis=0)[j] > best[0]:
            best = (f1.mean(axis=0)[j], lambda_, THRESHOLDS[j])
    best_f1, best_lambda, best_threshold = best
    print(f"Selected lambda={best_lambda}, threshold={best_threshold:.2f} (CV F1 {best_f1:.4f})")

    # the fold states only existed to score the folds; the preprocessing applied to the
    # test set has to be fitted on the same rows the submitted weights are trained on
    weights, state = fit_fold(x_train, y_train, feature_names, lambdas=(best_lambda,))
    y_pred = predict_labels_logistic(weights[best_lambda], transform_features(x_test, state),
                                     best_threshold)
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"Wrote {args.output}: {int(np.sum(y_pred == 1))} of {len(y_pred)} rows predicted positive")


if __name__ == "__main__":
    main()
