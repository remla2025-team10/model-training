import joblib
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

if __name__ == "__main__":
    bow_path = 'models/BoW_Sentiment_Model.pkl'
    classifier_path = 'models/Classifier_Sentiment_Model'
    
    vectorizer, classifier = load_models(bow_path, classifier_path)
    
    reviews = ["Wow... Loved this place.", "Crust is not good.", "Not tasty and the texture was wrong."]
    predictions = predict(reviews, vectorizer, classifier)
    
    print(predictions)