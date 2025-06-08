import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from restaurant_model_training.modeling import train, predict
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

RAW_DATA_PATH = config.RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def model_setup(tmp_path_factory):
    """Fixture to prepare data, features, and train the model."""

    # define temporary files
    tmp_path = tmp_path_factory.mktemp("model_test")
    data_p = str(RAW_DATA_PATH)
    processed_p = tmp_path / "processed.csv"
    bow_p = tmp_path / "bow.pkl"
    model_p = tmp_path / "model.joblib"

    # get data and features
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)

    # train model
    model = train.train_model(features, labels, model_p, test_size=0.2, random_state=42)

    return features, labels, model, model_p, bow_p

# Model 1, Infra 1: training determinism, config consistency
def test_model_reproducibility(model_setup, tmp_path):
    """Test that model training is reproducible."""
    features, labels, _, _, _ = model_setup

    # train two models with same params
    model_p1 = tmp_path / "model1.joblib"
    model_p2 = tmp_path / "model2.joblib"
    model1 = train.train_model(features, labels, model_p1, test_size=0.2, random_state=42)
    model2 = train.train_model(features, labels, model_p2, test_size=0.2, random_state=42)

    # check if identical (random state is the same, deterministic)
    pred1 = model1.predict(features)
    pred2 = model2.predict(features)
    assert np.array_equal(pred1, pred2), "Models should produce identical predictions with same parameters"

# Model 3: hyperparameter behavior verification
def test_model_hyperparameters(model_setup, tmp_path):
    """Test that model hyperparameters are properly set and effective."""
    features, labels, _, model_p, _ = model_setup

    # train two models with different test sizes
    model1 = train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    model2 = train.train_model(features, labels, model_p, test_size=0.3, random_state=42)

    # check if hyperparameters are configured right
    assert hasattr(model1, 'classes_'), "Model should have classes_ attribute"
    assert hasattr(model1, 'class_prior_'), "Model should have class_prior_ attribute"
    assert len(model1.classes_) == 2, "Model should have 2 classes"

    # check if predictions are different
    pred1 = model1.predict(features)
    pred2 = model2.predict(features)

    assert len(pred1) == len(features), "Predictions should match input size"
    assert len(pred2) == len(features), "Predictions should match input size"
    assert not np.array_equal(pred1, pred2), "Model predictions should differ with different test sizes"

# Model 6: predictive quality threshold enforcement
def test_model_performance_metrics(model_setup):
    """Test that model meets minimum performance requirements."""
    features, labels, model, _, _ = model_setup

    # calculate performance metrics
    y_pred = model.predict(features)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)

    # make sure they meet a standard (thresholds)
    assert accuracy >= 0.7, f"Accuracy {accuracy} below threshold 0.7"
    assert precision >= 0.6, f"Precision {precision} below threshold 0.6"
    assert recall >= 0.6, f"Recall {recall} below threshold 0.6"
    assert f1 >= 0.6, f"F1 score {f1} below threshold 0.6"

# Model 7: model outputs and rationale availability (interpretable)
def test_model_interpretability(model_setup):
    """Test that model provides interpretable outputs."""
    features, _, model, _, _ = model_setup

    # check if model has feature importances/coefficients
    assert hasattr(model, 'predict_proba'), "Model should support probability predictions"
    assert hasattr(model, 'class_prior_'), "Model should provide class priors"

    # predict probs
    proba = model.predict_proba(features)

    # are probs well formed?
    assert proba.shape[1] == 2, "Should predict probabilities for both classes"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1"

# TESTS ON ROBUSTNESS BELOW

# non-determinism robustness
def test_model_non_determinism_robustness(model_setup):
    """Test model robustness to non-deterministic behavior using data slices."""
    features, _, model, _, _ = model_setup

    # split data in subsets (5 slices)
    slice_size = features.shape[0] // 5 
    predictions = []

    # for each slice, check if predictions are binary/consistent
    for i in range(5):
        data_slice = features[i * slice_size:(i + 1) * slice_size]
        pred = model.predict(data_slice)
        predictions.append(pred)
        assert all(p in [0, 1] for p in pred), "Predictions should be binary"

    # check if predictions in each slice are consistent
    for i in range(1, len(predictions)):
        assert len(predictions[i]) == len(predictions[i - 1]), "Prediction sizes across slices should match"

# check robustness: handling zeroed features. 
def test_model_robustness(model_setup):
    """Test model robustness to input variations."""
    features, _, model, _, _ = model_setup

    # check if the model can handle zeroed features
    for i in range(0, features.shape[1], 100):
        # step 1: zero out a subset/slice of features
        modified_features = features.copy()
        modified_features[:, i:i+100] = 0
        predictions = model.predict(modified_features)

        # step 2: check if predictions are still valid afterwards
        assert len(predictions) == modified_features.shape[0], "Predictions should match input size"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"