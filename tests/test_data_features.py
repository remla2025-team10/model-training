import pandas as pd
import numpy as np
from restaurant_model_training import dataset
import pytest

# basic dataset loading test using get_data()
def test_generate_features(raw_data_path, tmp_path):
    """basic dataset loading test using get_data()"""
    X, y = dataset.get_data(raw_data_path, processed_data_path=tmp_path / "processed.csv")
    assert X.shape[0] > 0, "Feature set should not be empty!"
    assert len(y) == X.shape[0], "Number of labels does not match number of reviews!"
    assert set(y).issubset({0, 1}), "Labels should only contain 0 or 1!"

# test that the 'liked' column is not heavily imbalanced
def test_liked_col_distribution(raw_data_path, tmp_path):
    """check that the 'liked' column is not heavily imbalanced"""
    _, y = dataset.get_data(raw_data_path, processed_data_path=tmp_path / "processed.csv")
    labels = np.unique(y)
    assert len(labels) > 1, "Labels should contain at least two classes"

    liked_counts = pd.Series(y).value_counts(normalize=True) # get the liked counts (liked = 1, disliked = 0)

    # check that no class has >90% of the data (imbalanced distribution)
    assert liked_counts.max() <= 0.9, "The 'liked' column is too imbalanced!"
    assert liked_counts.min() >= 0.1, "The 'liked' column is too imbalanced!"

@pytest.mark.ml_test_score(category_test="Data1", status="automatic")
# Data1: Feature expectations are captured in a schema. We assume the reviews must be non-empty strings!
def test_non_empty_reviews(raw_data_path):
    """check that raw review text entries are not empty (schema validation)"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3)
    assert df['Review'].str.strip().str.len().min() > 0, "Some reviews are empty!" # check for empty reviews

@pytest.mark.ml_test_score(category_test="Data5", status="automatic")
# Data5: the data pipeline has appropriate privacy controls
def test_no_emails_or_phones_in_reviews(raw_data_path):
    """heuristic test to identify personally identifiable information (PII) such as emails and phone numbers"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3) 
    assert not df['Review'].str.contains(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', regex=True).any(), "Emails detected!"
    assert not df['Review'].str.contains(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', regex=True).any(), "Phone numbers detected!"