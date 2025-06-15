"""
Makes an example prediction to test the model and show how it works.
"""
import argparse
import pickle
import joblib
import pandas as pd
from preprocess_sentiment_analysis import preprocess_dataframe
from .. import config

def load_models(bow_path, classifier_path):
    """Load the vectorizer and classifier models."""
    with open(bow_path, "rb") as f:
        vectorizer = pickle.load(f)
    classifier = joblib.load(classifier_path)
    return vectorizer, classifier

def predict(input_reviews, input_vectorizer, input_classifier,
            output_path=f'{config.PROCESSED_DATA_DIR}/processed_test_reviews.csv'):
    """Predict sentiment of reviews using pre-trained models."""
    df = pd.DataFrame(input_reviews, columns=['Review'])
    df = preprocess_dataframe(df, output_path=output_path)

    processed_reviews = pd.read_csv(output_path)
    corpus = processed_reviews['Processed_Review']
    features = input_vectorizer.transform(corpus).toarray()

    prediction_results = input_classifier.predict(features)

    return prediction_results

def create_argument_parser():
    """Create argument parser for prediction script."""
    parser = argparse.ArgumentParser(
        description='Predict sentiment of reviews using a pre-trained model.')
    parser.add_argument(
        '--bow_p', type=str, default=str(config.DEFAULT_BOW_MODEL_PATH),
        help=f'Path of the BOW model (default: '
             f'{config.DEFAULT_BOW_MODEL_PATH})')
    parser.add_argument(
        '--model_p', type=str, default=str(config.DEFAULT_CLASSIFIER_MODEL_PATH),
        help=f'Path to save the trained model (default: '
             f'{config.DEFAULT_CLASSIFIER_MODEL_PATH})')
    return parser

if __name__ == "__main__":
    argument_parser = create_argument_parser()
    args = argument_parser.parse_args()

    bow_model_path = args.bow_p
    classifier_model_path = args.model_p

    model_vectorizer, model_classifier = load_models(bow_model_path, classifier_model_path)

    test_reviews = ["Wow... Loved this place.", "Crust is not good.",
                    "Not tasty and the texture was wrong."]
    sentiment_predictions = predict(test_reviews, model_vectorizer, model_classifier)

    print(sentiment_predictions)
    