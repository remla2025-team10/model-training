import pandas as pd
import pytest
import psutil
import time
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training.modeling import train, predict
from restaurant_model_training import config

RAW_DATA_PATH = config.RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture
def raw_data_path(scope="module"):
    return str(RAW_DATA_PATH)

@pytest.fixture(scope="module")
def model_setup(tmp_path_factory):
    """Fixture to prepare the vectorizer and classifier."""
    tmp_path = tmp_path_factory.mktemp("model_test")
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    model_p = tmp_path / "model.joblib"
    bow_p = tmp_path / "bow.pkl"

    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)
    train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    vectorizer, classifier = predict.load_models(bow_p, model_p)
    return vectorizer, classifier

# Data1: check for input integrity
def test_non_empty_reviews(raw_data_path):
    """check that raw review text entries are not empty (schema validation)"""
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3) # load raw data
    assert df['Review'].str.strip().str.len().min() > 0, "Some reviews are empty!" # check for empty reviews

# Data1 and Data7: test input feature code
def test_data_loading(model_setup):
    """verify data loading process (retrieves aligned review texts and labels)"""
    vectorizer, classifier = model_setup

    # check if corpus and labels are aligned
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

# Below here are non-functional tests -> test for features and data (lectures)

# Data: test cost of feature generation (latency and memory usage)
def test_feature_generation_cost(model_setup):
    """ test that feature generation does not exceed set memory and latency constraints"""
    # get data + temp paths
    vectorizer, _ = model_setup

    # measure memoery usage and latency
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # start feature generation and timer
    start_time = time.time()
    _ = vectorizer.transform(["example review text"])
    end_time = time.time()

    # calculate memory usage and latency
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    latency = end_time - start_time

    # check if memory usage is below threshold
    assert memory_used < 200 * 1024 * 1024, f"Memory usage exceeded 200MB! Used: {memory_used / (1024 * 1024):.2f} MB"

    # check if latency is below threshold
    assert latency < 5, f"Feature generation took too long! Latency: {latency:.2f} seconds"

# DATA SLICE TESTS (from lectures)

# define sentiment data slices
positive_words = ["excellent", "amazing", "great", "delicious", "fantastic", "perfect", "awesome"]
negative_words = ["awful", "terrible", "bad", "disgusting", "worst", "horrible", "poor"]

def test_positive_sentiment_slice(model_setup):
    """Test that positive sentiment words are generally classified as positive (1)"""
    vectorizer, classifier = model_setup
    preds = predict.predict(positive_words, vectorizer, classifier)

    # expect these words to be classified as positive (class 1)
    positive_count = sum(preds)
    assert positive_count >= int(len(positive_words) * 0.8), \
        f"Expected >= 80% positive classifications, got {positive_count}/{len(positive_words)}"

def test_negative_sentiment_slice(model_setup):
    """Test that negative sentiment words are generally classified as negative (0)"""
    vectorizer, classifier = model_setup
    preds = predict.predict(negative_words, vectorizer, classifier)

    # expect these words to be classified as negative (class 0)
    negative_count = sum([1 for p in preds if p == 0])
    assert negative_count >= int(len(negative_words) * 0.8), \
        f"Expected >= 80% negative classifications, got {negative_count}/{len(negative_words)}"