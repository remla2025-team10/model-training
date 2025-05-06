import pickle
import joblib
import argparse
import pandas as pd
from preprocess_sentiment_analysis import preprocess_dataframe
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def train(data_p, processed_p, bow_p, model_p, bow_max_features):
    """
    Trains a sentiment analysis model using a Naive Bayes classifier.

    Args:
        data_p (str): Path to the input training data file (TSV format).
        processed_p (str): Path to save the processed reviews CSV file.
        bow_p (str): Path to save the Bag-of-Words (CountVectorizer) model (as a pickle file).
        model_p (str): Path to save the trained sentiment classifier (as a joblib file).
        bow_max_features (int): Maximum number of features for the Bag-of-Words vectorizer.

    Returns:
        None
    """
    dataset = pd.read_csv(data_p, delimiter='\t', quoting=3)
    corpus = preprocess_dataframe(dataset, output_path=str(processed_p))['Processed_Review']
    cv = CountVectorizer(max_features=bow_max_features)

    X = cv.fit_transform(corpus).toarray()
    y = dataset.loc[:, 'Liked'].values

    # Saving BoW dictionary to later use in prediction
    pickle.dump(cv, open(bow_p, "wb"))

    # Dividing dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model fitting (Naive Bayes)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Model performance
    y_pred = classifier.predict(X_test)

    print(f'Model accuracy: {accuracy_score(y_test, y_pred)}')

    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, model_p)

    return


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Train a sentiment analysis model with specified data, model, and output paths, and configure the Bag-of-Words vectorizer.')
    parser.add_argument('--data_p', type=str, default='data/a1_RestaurantReviews_HistoricDump.tsv',
                        help='Path of the training data (default: data/a1_RestaurantReviews_HistoricDump.tsv)')
    parser.add_argument('--model_p', type=str, default=f'model/Classifier_Sentiment_Model',
                        help=f'Path to save the trained model (default: model/Classifier_Sentiment_Model)')
    parser.add_argument('--bow_p', type=str, default=f'model/BoW_Sentiment_Model.pkl',
                        help=f'Path to save the BoW dictionary (default: model/BoW_Sentiment_Model.pkl)')
    parser.add_argument('--processed_p', type=str, default='data/processed_reviews.csv',
                        help='Path to save the processed reviews (default: data/processed_reviews.csv)')
    parser.add_argument('--bow_max_features', type=int, default=1420,
                        help='Count Vectorizer max features (default: 1420)')
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    train(
        data_p=args.data_p,
        processed_p=args.processed_p,
        bow_p=args.bow_p,
        model_p=args.model_p,
        bow_max_features=args.bow_max_features
    )
