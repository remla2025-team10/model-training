import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from restaurant_model_training.modeling import train
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

def prepare_data_and_features(tmp_path, test_size=0.2, random_state=42, model_p=None):
    """Helper to load data, create features, and train model."""

    # define paths
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p = tmp_path / "processed.csv"
    bow_p = tmp_path / "bow.pkl"

    # get data and features
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)

    # train the model
    model = train.train_model(features, labels, model_p, test_size=test_size, random_state=random_state)
    
    # return features, labels, model
    return features, labels, model

# Model 6: predictive quality threshold enforcement
def test_model_performance_metrics(tmp_path):
    """Test that model meets min performance requirements."""

    # define paths
    model_p = tmp_path / "model.joblib"
    features, labels, model = prepare_data_and_features(tmp_path, model_p=model_p)

    # metrics
    y_pred = model.predict(features)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)

    # check if metrics meet min thresholds
    assert accuracy >= 0.7, f"Accuracy {accuracy} below threshold 0.7"
    assert precision >= 0.6, f"Precision {precision} below threshold 0.6"
    assert recall >= 0.6, f"Recall {recall} below threshold 0.6"
    assert f1 >= 0.6, f"F1 score {f1} below threshold 0.6"

# Model 3: hyperparameter behavior verification
def test_model_hyperparameters(tmp_path):
    """Test that model hyperparameters are properly set and effective."""

    # define paths
    model_p = tmp_path / "model.joblib"
    features, labels, _ = prepare_data_and_features(tmp_path, model_p=model_p)

    # train models with different hyperparameters
    model1 = train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    model2 = train.train_model(features, labels, model_p, test_size=0.3, random_state=42)

    # check if hyperparameters are set correctly
    assert hasattr(model1, 'classes_'), "Model should have classes_ attribute"
    assert hasattr(model1, 'class_prior_'), "Model should have class_prior_ attribute"
    assert len(model1.classes_) == 2, "Model should have 2 classes"

    # check if predictions differ with different test sizes
    pred1 = model1.predict(features)
    pred2 = model2.predict(features)

    assert len(pred1) == len(features), "Predictions should match input size"
    assert len(pred2) == len(features), "Predictions should match input size"
    assert not np.array_equal(pred1, pred2), "Model predictions should differ with different test sizes"

# Model 5: simple is not always better (check robustness)
def test_model_robustness(tmp_path):
    """Test model robustness to input variations."""

    # define paths
    model_p = tmp_path / "model.joblib"
    features, _ , model = prepare_data_and_features(tmp_path, model_p=model_p)

    # check if model can handle zero-ing out features
    for i in range(0, features.shape[1], 100):

        # zero out a subset of features
        modified_features = features.copy()
        modified_features[:, i:i+100] = 0
        predictions = model.predict(modified_features)

        # check if predictions are still binary
        assert len(predictions) == modified_features.shape[0], "Predictions should match input size"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"

# Model 1, Infra 1: training determinism, config consistency
def test_model_reproducibility(tmp_path):
    """Test that model training is reproducible."""

    # define paths
    model_p1 = tmp_path / "model1.joblib"
    model_p2 = tmp_path / "model2.joblib"
    features, labels, _ = prepare_data_and_features(tmp_path, model_p=tmp_path / "model.joblib")

    # train two models with same parameters
    model1 = train.train_model(features, labels, model_p1, test_size=0.2, random_state=42)
    model2 = train.train_model(features, labels, model_p2, test_size=0.2, random_state=42)
    
    # check if models are identical
    pred1 = model1.predict(features)
    pred2 = model2.predict(features)
    assert np.array_equal(pred1, pred2), "Models should produce identical predictions with same parameters"

# Model 7: model outputs and rationale availability
def test_model_interpretability(tmp_path):
    """Test that model provides interpretable outputs."""

    # define paths
    model_p = tmp_path / "model.joblib"
    features, _ , model = prepare_data_and_features(tmp_path, model_p=model_p)

    # check if model has feature importances/coefficients
    assert hasattr(model, 'predict_proba'), "Model should support probability predictions"
    assert hasattr(model, 'class_prior_'), "Model should provide class priors"

    # predict probabilities
    proba = model.predict_proba(features)

    # check if probabilities are well-formed
    assert proba.shape[1] == 2, "Should predict probabilities for both classes"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1"