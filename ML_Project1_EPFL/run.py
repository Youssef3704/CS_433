"""Reproducible training and submission pipeline stubs."""
import time

from ML_Project1_EPFL.helpers import (load_csv_data, load_feature_names,
                                      predict_labels_logistic, create_csv_submission)
from ML_Project1_EPFL.implementations import logistic_regression, reg_logistic_regression, sigmoid
from ML_Project1_EPFL.preprocessing import fit_preprocessor, transform_features, encode_labels
import numpy as np

from ML_Project1_EPFL.validation import split_data, compute_accuracy, compute_f1, cross_validate


def train_model(y, tx):
    """Train the selected model with the chosen hyperparameters and return weights."""



def generate_predictions(x_test, weights, preprocessing_state, threshold=0.5):
    """Transform test features and return competition labels {-1, 1}."""
    tx_test = transform_features(x_test, preprocessing_state)
    return predict_labels_logistic(weights, tx_test, threshold)


def main():
    """Load data, fit preprocessing, train the model, and write the submission CSV."""
    print("Started")

    #load data
    x_train_ai_crowd, x_test_aicrowd, y_train_ai_crowd, train_ids, test_ids = load_csv_data("data/81fdc2af-6811-4089-9054-9d59f417f819_dataset/dataset")
    # column names, so the preprocessor can drop the survey metadata by name
    feature_names = load_feature_names("data/81fdc2af-6811-4089-9054-9d59f417f819_dataset/dataset")

    x_train, x_test, y_train, y_test = split_data(x_train_ai_crowd, y_train_ai_crowd, 1)

    # the logistic gradient expects labels in {0, 1}, not the {-1, 1} of the competition
    y_train = encode_labels(y_train)

    #now let's do some cross validation with regularized logistic regression seeing lambdas and thresholds

    lambdas = [0.0,0.1,0.2,0.5,0.7,0.9,0.95,1.0,1.05,1.1]
    thresholds = [0.1,0.15,0.2,0.3,0.35,0.4,0.42,0.45,0.5]

    f1_scores_array=np.zeros(shape=(len(lambdas),len(thresholds)))

    for i in range(len(lambdas)):
        # the threshold plays no part in training, so fit each fold once and
        # score the stored probabilities at every threshold
        start = time.time()
        folds = cross_validate(x_train, y_train,
                               lambda x, y: _fit_logistic_ridge(x, y, lambdas[i], feature_names))
        print(f"lambda={lambdas[i]} ({i + 1}/{len(lambdas)}): "
              f"{5} folds fitted in {time.time() - start:.0f}s", flush=True)

        validation_probabilities = []
        for fold in folds.values():
            weights, preprocessing_state = fold["model"]
            valid_idx = fold["valid_idx"]
            tx_valid = transform_features(x_train[valid_idx], preprocessing_state)
            validation_probabilities.append((y_train[valid_idx], sigmoid(tx_valid @ weights)))

        for j in range(len(thresholds)):
            f1_scores_array[i,j]=np.mean([compute_f1(y_true, np.where(probabilities >= thresholds[j], 1, 0))
                                          for y_true, probabilities in validation_probabilities])

    maximum = np.unravel_index(np.argmax(f1_scores_array, axis=None), f1_scores_array.shape)
    best_lambda = lambdas[maximum[0]]
    best_threshold = thresholds[maximum[1]]

    print(f"Cross validated best parameters lambda={best_lambda}, threshold={best_threshold}")
    print(f"Best f1_score={f1_scores_array[maximum[0],maximum[1]]}")

    # the fold states only existed to score the folds; the preprocessing applied to the
    # test set has to be fitted on the same rows the submitted weights are trained on
    print("Refitting on the full training set", flush=True)
    start = time.time()
    weights, preprocessing_state = _fit_logistic_ridge(x_train, y_train, best_lambda, feature_names)
    print(f"Refitted in {time.time() - start:.0f}s", flush=True)

    y_pred = generate_predictions(x_test_aicrowd, weights, preprocessing_state, best_threshold)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"Wrote submission.csv: {int(np.sum(y_pred == 1))} of {len(y_pred)} rows predicted positive")


def _fit_logistic_ridge(x, y, lambda_, feature_names=None, max_iters=500, gamma=0.1):
    """Fit preprocessing and regularized logistic regression on one training fold."""
    preprocessing_state = fit_preprocessor(x, feature_names)
    tx = transform_features(x, preprocessing_state)
    weights, _ = reg_logistic_regression(y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, gamma)
    return weights, preprocessing_state


def _simple_logistic_regression(x_train, y_train, x_test, y_test):
    # we need to split the train and test set into our own sets

    preprocessing_state = fit_preprocessor(x_train)

    x_train_transformed = transform_features(x_train, preprocessing_state)
    y_train_transformed = encode_labels(y_train)

    x_test_transformed = transform_features(x_test, preprocessing_state)
    y_test_transformed = encode_labels(y_test)

    print("Loaded data and starting")

    # start with plain old logistic regression
    weights, loss = logistic_regression(y_train_transformed, x_train_transformed,
                                        np.zeros(x_train_transformed.shape[1]), 500, 0.1)

    # do predictions

    y_train_pred = encode_labels(predict_labels_logistic(weights, x_train_transformed))
    y_test_pred = encode_labels(predict_labels_logistic(weights, x_test_transformed))

    accuracy_train = compute_accuracy(y_train_transformed, y_train_pred)
    f1_score_train = compute_f1(y_train_transformed, y_train_pred)

    print(f"Accuracy train: {accuracy_train}")
    print(f"F1 score train: {f1_score_train}")

    accuracy_test = compute_accuracy(y_test_transformed, y_test_pred)
    f1_score_test = compute_f1(y_test_transformed, y_test_pred)

    print(f"Accuracy test: {accuracy_test}")
    print(f"F1 score test: {f1_score_test}")

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        "data/81fdc2af-6811-4089-9054-9d59f417f819_dataset/dataset")

    state = fit_preprocessor(x_train)
    tx_train = transform_features(x_train, state)
    tx_test = transform_features(x_test, state)

    weights, _ = logistic_regression(encode_labels(y_train), tx_train,
                                     np.zeros(tx_train.shape[1]), 500, 0.1)

    y_pred = predict_labels_logistic(weights, tx_test, threshold=0.15)  # values in {-1, 1}
    create_csv_submission(test_ids, y_pred, "submission.csv")


if __name__ == "__main__":
    main()
