import pickle
from sklearn.feature_extraction.text import CountVectorizer

def create_bow_features(corpus, max_features, bow_path):
    vectorizer = CountVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(corpus).toarray()
    pickle.dump(vectorizer, open(bow_path, "wb"))
    
    return features