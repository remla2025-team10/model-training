"""
Train a sentiment analysis model for restaurant reviews.
This module also includes argument parsing for specifying parameter values via command line.
"""
from pathlib import Path
import argparse
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from restaurant_model_training import config
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features

def train_model(input_features, input_labels, model_output_path,
                test_size=config.DEFAULT_TEST_SIZE,
                random_state=config.DEFAULT_RANDOM_STATE):
    """Train a sentiment analysis model."""
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, input_labels, test_size=test_size, random_state=random_state)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    print(f'Test size: {test_size}')
    print(f'Random state: {random_state}')

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(classifier, model_output_path)
    print(f'Model accuracy: {accuracy}')

    # More metrics
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Store accuracy in a JSON file (for DVC)
    with open(config.DEFAULT_METRICS_PATH, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
        }, f, indent=2)

    return classifier

def create_argument_parser():
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(
        description='Train a sentiment analysis model with specified data, '
                    'model, and output paths, and configure the Bag-of-Words '
                    'vectorizer.')
    parser.add_argument(
        '--data_p', type=str, default=str(config.DEFAULT_RAW_DATA_PATH),
        help=f'Path of the training data (default: '
             f'{config.DEFAULT_RAW_DATA_PATH})')
    parser.add_argument(
        '--model_p', type=str, default=str(config.DEFAULT_CLASSIFIER_MODEL_PATH),
        help=f'Path to save the trained model (default: '
             f'{config.DEFAULT_CLASSIFIER_MODEL_PATH})')
    parser.add_argument(
        '--bow_p', type=str, default=str(config.DEFAULT_BOW_MODEL_PATH),
        help=f'Path to save the BoW dictionary (default: '
             f'{config.DEFAULT_BOW_MODEL_PATH})')
    parser.add_argument(
        '--processed_p', type=str,
        default=str(config.DEFAULT_PROCESSED_DATA_PATH),
        help=f'Path to save the processed reviews (default: '
             f'{config.DEFAULT_PROCESSED_DATA_PATH})')
    parser.add_argument(
        '--bow_max_features', type=int, default=1420,
        help=f'Count Vectorizer max features (default: '
             f'{config.DEFAULT_MAX_FEATURES})')
    parser.add_argument(
        '--test_size', type=float, default=config.DEFAULT_TEST_SIZE,
        help=f'Test size for the training: (default: '
             f'{config.DEFAULT_TEST_SIZE})')
    parser.add_argument(
        '--random_state', type=int, default=config.DEFAULT_RANDOM_STATE,
        help=f'Random state for the training: (default: '
             f'{config.DEFAULT_RANDOM_STATE})')
    return parser

if __name__ == "__main__":
    argument_parser = create_argument_parser()
    args = argument_parser.parse_args()

    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.METRICS_DIR).mkdir(parents=True, exist_ok=True)

    corpus, data_labels = get_data(
        raw_data_path=args.data_p,
        processed_data_path=args.processed_p
    )

    feature_vectors = create_bow_features(
        corpus=corpus,
        max_features=args.bow_max_features,
        bow_path=args.bow_p
    )

    train_model(
        input_features=feature_vectors,
        input_labels=data_labels,
        model_output_path=args.model_p,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Competed training and saved to {args.model_p}")
