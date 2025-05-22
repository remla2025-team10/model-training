import joblib
import argparse
import pandas as pd
import pickle
from preprocess_sentiment_analysis import preprocess_dataframe
from .. import config

def load_models(bow_path, classifier_path):
    vectorizer = pickle.load(open(bow_path, "rb"))
    classifier = joblib.load(classifier_path)
    return vectorizer, classifier

def predict(reviews, vectorizer, classifier, output_path=f'{config.PROCESSED_DATA_DIR}/processed_test_reviews.csv'):

    df = pd.DataFrame(reviews, columns=['Review'])
    df = preprocess_dataframe(df, output_path=output_path)

    processed_reviews = pd.read_csv(output_path)
    corpus = processed_reviews['Processed_Review']
    features = vectorizer.transform(corpus).toarray()

    predictions = classifier.predict(features)

    return predictions

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Predict sentiment of reviews using a pre-trained model.')
    parser.add_argument('--bow_p', type=str, default=str(config.DEFAULT_BOW_MODEL_PATH),
                        help=f'Path of the BOW model (default: {config.DEFAULT_BOW_MODEL_PATH})')
    parser.add_argument('--model_p', type=str, default=str(config.DEFAULT_CLASSIFIER_MODEL_PATH),
                        help=f'Path to save the trained model (default: {config.DEFAULT_CLASSIFIER_MODEL_PATH})')
    return parser

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    bow_path = args.bow_p
    classifier_path = args.model_p
    
    vectorizer, classifier = load_models(bow_path, classifier_path)
    
    reviews = ["Wow... Loved this place.", "Crust is not good.", "Not tasty and the texture was wrong."]
    predictions = predict(reviews, vectorizer, classifier)
    
    print(predictions)