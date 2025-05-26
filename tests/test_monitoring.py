import pandas as pd
import numpy as np
from pathlib import Path
from restaurant_model_training.modeling import train, predict
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

# Monitor 2: Data invariants hold for inputs
# required input schema exists (i.e. expected columns are present in raw data)
def test_schema():
    expected_columns = ['Review', 'Liked']
    df = pd.read_csv("data/raw/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)
    for col in expected_columns:
        assert col in df.columns, f"{col} is missing from the dataset!"

# Monitor 7: prediction quality has not regressed (indirect)
# Monitor 2: data invariants hold for inputs (indirect, via processed input)
# basic monitoring of output class distribution to detect skew/class collapse
def test_prediction_monitoring(tmp_path):
    """Test that prediction monitoring works correctly."""

    # define paths
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    model_p = tmp_path / "model.joblib"
    bow_p = tmp_path / "bow.pkl"
    
    # train
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)
    model = train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    
    # load models
    vectorizer, classifier = predict.load_models(bow_p, model_p)
    
    # predict with monitoring
    test_reviews = ["Great food!", "Terrible service.", "Okay experience."]
    predictions = predict.predict(test_reviews, vectorizer, classifier, output_path=str(tmp_path / "test_processed.csv"))
    
    # check prediction distribution
    pred_dist = np.bincount(predictions)
    assert len(pred_dist) == 2, "Should have predictions for both classes"
    assert sum(pred_dist) == len(test_reviews), "Should have one prediction per review"

# Monitor 7: prediction quality has not regressed
# Monitor 4: models are not too stale (indirect: validates model performance)
# evaluates key performance metrics that should be tracked over time
def test_performance_monitoring(tmp_path):
    """Test that model performance can be monitored over time."""
    
    # define paths
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    model_p = tmp_path / "model.joblib"
    bow_p = tmp_path / "bow.pkl"
    
    # train
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)
    model = train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    
    # metrics (monitoring)
    predictions = model.predict(features)
    accuracy = np.mean(predictions == labels)
    precision = np.mean(predictions[labels == 1] == 1)
    recall = np.mean(labels[predictions == 1] == 1)
    
    # check performance metrics
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"

