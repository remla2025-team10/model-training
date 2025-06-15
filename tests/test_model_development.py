"""
Test for model development and performance metrics.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytest

from restaurant_model_training.modeling import train, predict

@pytest.mark.ml_test_score(category_test="Model6", status="automatic")
# Model 6: predictive quality threshold enforcement
def test_model_performance_metrics(model_setup, threshold):
    """Test that model meets minimum performance requirements."""
    features, labels, model, _, _ = model_setup

    # calculate performance metrics
    y_pred = model.predict(features)
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)

    # make sure they meet a standard (thresholds)
    assert accuracy >= threshold, f"Accuracy {accuracy} below threshold {threshold}"
    assert precision >= threshold, f"Precision {precision} below threshold {threshold}"
    assert recall >= threshold, f"Recall {recall} below threshold {threshold}"
    assert f1 >= threshold, f"F1 score {f1} below threshold {threshold}"

    # Checking important data-slices

    # performance metrics on negative class
    y_pred_neg = y_pred[labels == 0]
    accuracy_neg = accuracy_score(labels[labels == 0], y_pred_neg)
    assert accuracy_neg >= threshold, (
        f"Negative class accuracy {accuracy_neg} below threshold {threshold}"
    )

    # performance metrics on positive class
    y_pred_pos = y_pred[labels == 1]
    accuracy_pos = accuracy_score(labels[labels == 1], y_pred_pos)
    assert accuracy_pos >= threshold, (
        f"Positive class accuracy {accuracy_pos} below threshold {threshold}"
    )


positive_words = ["excellent", "amazing", "great", "delicious", "fantastic", "perfect", "awesome"]
negative_words = ["awful", "terrible", "bad", "disgusting", "worst", "horrible", "poor"]

@pytest.mark.ml_test_score(category_test="Model6", status="automatic")
# Model 6: Sentiment analysis slice tests
def test_positive_sentiment_slice(model_setup):
    """Test that positive sentiment words are generally classified as positive (1)"""
    _, _, _, model_p, bow_p = model_setup
    vectorizer, classifier = predict.load_models(bow_p, model_p)

    preds = predict.predict(positive_words, vectorizer, classifier)

    # expect these words to be classified as positive (class 1)
    positive_count = sum(preds)
    assert positive_count >= int(len(positive_words) * 0.8), \
        f"Expected >= 80% positive classifications, got {positive_count}/{len(positive_words)}"

@pytest.mark.ml_test_score(category_test="Model6", status="automatic")
def test_negative_sentiment_slice(model_setup):
    """Test that negative sentiment words are generally classified as negative (0)"""
    _, _, _, model_p, bow_p = model_setup
    vectorizer, classifier = predict.load_models(bow_p, model_p)

    preds = predict.predict(negative_words, vectorizer, classifier)

    # expect these words to be classified as negative (class 0)
    negative_count = sum(p == 0 for p in preds)
    assert negative_count >= int(len(negative_words) * 0.8), \
        f"Expected >= 80% negative classifications, got {negative_count}/{len(negative_words)}"

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
        assert len(predictions[i]) == len(predictions[i - 1]), (
            "Prediction sizes across slices should match"
        )

# check robustness: handling zeroed features.
def test_model_robustness(model_setup):
    """Test model robustness to input variations."""
    features, _, model, _, _ = model_setup

    # check if the model can handle zeroed features
    for i in range(0, features.shape[1], 100):
        # step 1: zero out a subset/slice of features
        modified_features = features.copy()
        modified_features[:, i:i + 100] = 0
        predictions = model.predict(modified_features)

        # step 2: check if predictions are still valid afterwards
        assert len(predictions) == modified_features.shape[0], "Predictions should match input size"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"

@pytest.mark.ml_test_score(category_test="Model3", status="automatic")
# Model predictions should differ with different test sizes
def test_model_hyperparameters(model_setup):
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
    assert not np.array_equal(pred1, pred2), (
        "Model predictions should differ with different test sizes"
    )