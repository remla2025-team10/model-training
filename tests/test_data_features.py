import pandas as pd
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features

# Data1: check for input integrity
# check that raw review text entries are not empty (schema validation)
def test_non_empty_reviews():
    df = pd.read_csv("data/raw/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)
    assert df['Review'].str.strip().str.len().min() > 0, "Some reviews are empty!"

# Data1 and Data7: test input feature code
# verify data loading process (retrieves aligned review texts and labels)
def test_data_loading(tmp_path):
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    corpus, labels = get_data(raw_data_path=data_p, processed_data_path=processed_p)
    assert len(corpus) == len(labels)
    assert all(isinstance(text, str) for text in corpus)

# Data1 and Data7: validates feature generation and tests BoW integrity
# check that feature extraction does not exceed specified constraints (e.g, max_features=100)
def test_feature_vector_shape(tmp_path):
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    bow_p = tmp_path / "bow.pkl"
    corpus, _ = get_data(raw_data_path=data_p, processed_data_path=processed_p)
    features = create_bow_features(corpus=corpus, max_features=100, bow_path=bow_p)
    assert features.shape[1] <= 100

# Data5: the data pipeline has appropriate privacy controls
# heuristic test to identify personally identifiable information (PII) such as emails and phone numbers
def test_no_emails_or_phones_in_reviews():
    df = pd.read_csv("data/raw/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)
    assert not df['Review'].str.contains(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', regex=True).any(), "Emails detected!"
    assert not df['Review'].str.contains(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', regex=True).any(), "Phone numbers detected!"
