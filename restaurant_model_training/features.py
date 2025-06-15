"""
Create Bag of Words features from a corpus of text data.
"""
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def create_bow_features(corpus, max_features, bow_path):
    """
    Create Bag of Words features from the given corpus and save it to the specified path.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(corpus).toarray()
    with open(bow_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return features
