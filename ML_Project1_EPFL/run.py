"""Reproducible training and submission pipeline stubs."""
from ML_Project1_EPFL.helpers import load_csv_data, predict_labels
from ML_Project1_EPFL.implementations import logistic_regression
from ML_Project1_EPFL.preprocessing import fit_preprocessor, transform_features, encode_labels
import numpy as np

from ML_Project1_EPFL.validation import split_data, compute_accuracy, compute_f1


def train_model(y, tx):
    """Train the selected model with the chosen hyperparameters and return weights."""



def generate_predictions(x_test, weights, preprocessing_state):
    """Transform test features and return competition labels {-1, 1}."""
    pass


def main():
    """Load data, fit preprocessing, train the model, and write the submission CSV."""
    print("Started")
    #load data
    x_train_ai_crowd, x_test_aicrowd, y_train_ai_crowd, train_ids, test_ids = load_csv_data("data/81fdc2af-6811-4089-9054-9d59f417f819_dataset/dataset")


    #we need to split the train and test set into our own sets
    x_train, x_test, y_train, y_test = split_data(x_train_ai_crowd, y_train_ai_crowd, 0.8)

    preprocessing_state = fit_preprocessor(x_train)

    x_train_transformed = transform_features(x_train, preprocessing_state)
    y_train_transformed = encode_labels(y_train)

    x_test_transformed = transform_features(x_test, preprocessing_state)
    y_test_transformed = encode_labels(y_test)

    print("Loaded data and starting")

    #start with plain old logistic regression
    weights, loss = logistic_regression(y_train_transformed, x_train_transformed, np.zeros(x_train_transformed.shape[1]), 500, 0.1)

    #do predictions

    y_train_pred = encode_labels(predict_labels(weights, x_train_transformed))
    y_test_pred = encode_labels(predict_labels(weights, x_test_transformed))

    accuracy_train = compute_accuracy(y_train_transformed, y_train_pred)
    f1_score_train = compute_f1(y_train_transformed, y_train_pred)

    print(f"Accuracy train: {accuracy_train}")
    print(f"F1 score train: {f1_score_train}")

    accuracy_test = compute_accuracy(y_test_transformed, y_test_pred)
    f1_score_test = compute_f1(y_test_transformed, y_test_pred)

    print(f"Accuracy test: {accuracy_test}")
    print(f"F1 score test: {f1_score_test}")




if __name__ == "__main__":
    main()
