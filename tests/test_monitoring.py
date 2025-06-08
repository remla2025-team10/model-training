import pandas as pd
import numpy as np
from restaurant_model_training.modeling import train, predict
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

# Monitor 2: Data invariants hold for inputs
def test_schema(raw_data_path):
    """Test that required columns exist (i.e. expected columns are present in raw data)"""
    expected_columns = ['Review', 'Liked']
    df = pd.read_csv(raw_data_path, delimiter="\t", quoting=3)
    for col in expected_columns:
        assert col in df.columns, f"{col} is missing from the dataset!"

# Monitor 2, 7: basic monitoring of output class distribution to detect skew/class collapse
# (2) Data invariants hold in training and serving inputs
# (7) The model has not experienced a regression in prediction quality on served data
def test_prediction_monitoring(tmp_path, raw_data_path):
    """Test that prediction monitoring works correctly and predictions are distributed across classes."""

    # define paths
    data_p = raw_data_path
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
    assert len(pred_dist) == 2, "Should have predictions for both classes" # (2)
    assert sum(pred_dist) == len(test_reviews), "Should have one prediction per review" # (7)

# Monitor 4, 7: evaluates key performance metrics that are compared over time
# (4) Monitor 4: Models are not too stale (implicitly tested by performance monitoring)
# (7) The model has not experienced a regression in prediction quality on served data.
def test_performance_monitoring(tmp_path, raw_data_path):
    """Test that model performance does not regress over time (simulated)."""
    
    # get paths
    data_p = raw_data_path
    processed_p = tmp_path / "processed.csv"
    model_p_1 = tmp_path / "model_v1.joblib"
    model_p_2 = tmp_path / "model_v2.joblib"
    bow_p = tmp_path / "bow.pkl"
    
    # get data and features
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)
    
    # simulate baseline model (v1)
    model_v1 = train.train_model(features, labels, model_p_1, test_size=0.2, random_state=42)
    preds_v1 = model_v1.predict(features)
    acc_v1 = np.mean(preds_v1 == labels)
    prec_v1 = np.mean(preds_v1[labels == 1] == 1)
    rec_v1 = np.mean(labels[preds_v1 == 1] == 1)

    # simulate a retrained model (v2) with new seed (simulating new training data)
    model_v2 = train.train_model(features, labels, model_p_2, test_size=0.2, random_state=99)
    preds_v2 = model_v2.predict(features)
    acc_v2 = np.mean(preds_v2 == labels)
    prec_v2 = np.mean(preds_v2[labels == 1] == 1)
    rec_v2 = np.mean(labels[preds_v2 == 1] == 1)

    tolerance = 0.1  # 10% tolerance for metric drop

    # check for performance regression
    assert acc_v2 >= acc_v1 - tolerance, f"Accuracy regressed: {acc_v1:.2f} -> {acc_v2:.2f}"
    assert prec_v2 >= prec_v1 - tolerance, f"Precision regressed: {prec_v1:.2f} -> {prec_v2:.2f}"
    assert rec_v2 >= rec_v1 - tolerance, f"Recall regressed: {rec_v1:.2f} -> {rec_v2:.2f}"


