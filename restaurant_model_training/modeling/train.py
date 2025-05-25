import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pathlib import Path
import json

from restaurant_model_training import config
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features

def train_model(features, labels, model_output_path, test_size = config.DEFAULT_TEST_SIZE, random_state = config.DEFAULT_RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    print(f'Test size: {test_size}')
    print(f'Random state: {random_state}')

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(classifier, model_output_path)
    print(f'Model accuracy: {accuracy}')

    # Store accuracy in a JSON file (for DVC)
    with open(config.DEFAULT_METRICS_PATH, "w") as f:
        json.dump({"accuracy": accuracy}, f)

    return classifier

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Train a sentiment analysis model with specified data, model, and output paths, and configure the Bag-of-Words vectorizer.')
    parser.add_argument('--data_p', type=str, default=str(config.DEFAULT_RAW_DATA_PATH),
                        help=f'Path of the training data (default: {config.DEFAULT_RAW_DATA_PATH})')
    parser.add_argument('--model_p', type=str, default=str(config.DEFAULT_CLASSIFIER_MODEL_PATH),
                        help=f'Path to save the trained model (default: {config.DEFAULT_CLASSIFIER_MODEL_PATH})')
    parser.add_argument('--bow_p', type=str, default=str(config.DEFAULT_BOW_MODEL_PATH),
                        help=f'Path to save the BoW dictionary (default: {config.DEFAULT_BOW_MODEL_PATH})')
    parser.add_argument('--processed_p', type=str, default=str(config.DEFAULT_PROCESSED_DATA_PATH),
                        help=f'Path to save the processed reviews (default: {config.DEFAULT_PROCESSED_DATA_PATH})')
    parser.add_argument('--bow_max_features', type=int, default=1420,
                        help=f'Count Vectorizer max features (default: {config.DEFAULT_MAX_FEATURES})')
    parser.add_argument('--test_size', type=float, default=config.DEFAULT_TEST_SIZE,
                        help=f'Test size for the training: (default: {config.DEFAULT_TEST_SIZE})')
    parser.add_argument('--random_state', type=int, default=config.DEFAULT_RANDOM_STATE,
                        help=f'Random state for the training: (default: {config.DEFAULT_RANDOM_STATE})')
    return parser

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.METRICS_DIR).mkdir(parents=True, exist_ok=True)

    corpus, labels = get_data(
        raw_data_path=args.data_p,
        processed_data_path=args.processed_p
    )

    features = create_bow_features(
        corpus=corpus,
        max_features=args.bow_max_features,
        bow_path=args.bow_p
    )

    train_model(
        features=features,
        labels=labels,
        model_output_path=args.model_p,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Competed training and saved to {args.model_p}")