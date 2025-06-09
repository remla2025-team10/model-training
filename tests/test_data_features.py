import pandas as pd
import numpy as np
from restaurant_model_training import dataset, config

def test_generate_features(raw_data_path, tmp_path):
    """basic dataset loading test using get_data()"""
    X, y = dataset.get_data(raw_data_path, processed_data_path=tmp_path / "processed.csv")
    assert X.shape[0] > 0, "Feature set should not be empty!"
    assert len(y) == X.shape[0], "Number of labels does not match number of reviews!"
    assert set(y).issubset({0, 1}), "Labels should only contain 0 or 1!"

def test_liked_col_distribution(raw_data_path, tmp_path):
    """check that the 'liked' column is not heavily imbalanced"""
    _, y = dataset.get_data(raw_data_path, processed_data_path=tmp_path / "processed.csv")
    labels = np.unique(y)
    assert len(labels) > 1, "Labels should contain at least two classes"

    liked_counts = pd.Series(y).value_counts(normalize=True) # get the liked counts (liked = 1, disliked = 0)

    # check that no class has >90% of the data (imbalanced distribution)
    assert liked_counts.max() <= 0.9, "The 'liked' column is too imbalanced!"
    assert liked_counts.min() >= 0.1, "The 'liked' column is too imbalanced!"

# Data1: Feature expectations are captured in a schema. We assume the reviews must be non-empty strings!
def test_non_empty_reviews(raw_data_path):
    """check that raw review text entries are not empty (schema validation)"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3)
    assert df['Review'].str.strip().str.len().min() > 0, "Some reviews are empty!" # check for empty reviews

# Data1 and Data7: test input feature code
def test_data_loading(model_setup):
    """verify data loading process (make sure vectorizer and classifier are not None)"""
    vectorizer, classifier = model_setup

    # check if corpus and labels are not empty
    assert vectorizer is not None, "Vectorizer should not be None"
    assert classifier is not None, "Classifier should not be None"

# Data1 and Data7: validates feature generation and tests BoW integrity
def test_feature_vector_shape(model_setup):
    """check that feature extraction does not exceed specified constraints (e.g, max_features)"""
    vectorizer, _ = model_setup
    feature_names = vectorizer.get_feature_names_out()

    # check if features are below max_features limit
    assert len(feature_names) <= config.DEFAULT_MAX_FEATURES, "Feature vector exceeds max_features constraint"

# Data5: the data pipeline has appropriate privacy controls
def test_no_emails_or_phones_in_reviews(raw_data_path):
    """heuristic test to identify personally identifiable information (PII) such as emails and phone numbers"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3) 
    assert not df['Review'].str.contains(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', regex=True).any(), "Emails detected!"
    assert not df['Review'].str.contains(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', regex=True).any(), "Phone numbers detected!"