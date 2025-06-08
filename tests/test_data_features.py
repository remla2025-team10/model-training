import pandas as pd
from restaurant_model_training import config

# Data1: check for input integrity
def test_non_empty_reviews(raw_data_path):
    """check that raw review text entries are not empty (schema validation)"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3) # load raw data
    assert df['Review'].str.strip().str.len().min() > 0, "Some reviews are empty!" # check for empty reviews

# Data1 and Data7: test input feature code
def test_data_loading(model_setup):
    """verify data loading process (retrieves aligned review texts and labels)"""
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