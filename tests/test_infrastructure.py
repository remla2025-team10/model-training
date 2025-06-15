"""
Tests for the infrastructure of the restaurant model training.
"""
import subprocess
import argparse
import pickle
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pytest

from restaurant_model_training.modeling import train, predict
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

@pytest.mark.ml_test_score(category_test="Infra1", status="automatic")
# Infra 1: test reproducibility of training process
def test_reproducibility(tmp_path, raw_data_path):
    """Test that model training is reproducible with the same data and params"""

    # define temp paths
    data_p = raw_data_path
    processed_p1 = tmp_path / "proc1.csv"
    processed_p2 = tmp_path / "proc2.csv"
    model_p1 = tmp_path / "model1.joblib"
    model_p2 = tmp_path / "model2.joblib"
    bow_p1 = tmp_path / "bow1.pkl"
    bow_p2 = tmp_path / "bow2.pkl"

    # get data and features
    corpus1, labels1 = get_data(raw_data_path=data_p, processed_data_path=processed_p1)
    corpus2, labels2 = get_data(raw_data_path=data_p, processed_data_path=processed_p2)
    features1 = create_bow_features(
                    corpus=corpus1,
                    max_features=config.DEFAULT_MAX_FEATURES,
                    bow_path=bow_p1)
    features2 = create_bow_features(
                    corpus=corpus2,
                    max_features=config.DEFAULT_MAX_FEATURES,
                    bow_path=bow_p2)

    # run training twice with same seed and params
    train.train_model(features1, labels1, model_p1, test_size=0.2, random_state=100)
    train.train_model(features2, labels2, model_p2, test_size=0.2, random_state=100)

    # load models
    model1 = joblib.load(model_p1)
    model2 = joblib.load(model_p2)

    # same data, same seed -> same accuracy
    df = pd.read_csv(data_p, delimiter="\t", quoting=3)
    with open(bow_p1, 'rb') as f:
        X = pickle.load(f).transform(df['Review']).toarray()
    y = df['Liked']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    acc1 = accuracy_score(y_test, model1.predict(X_test))
    acc2 = accuracy_score(y_test, model2.predict(X_test))

    assert abs(acc1 - acc2) < 0.01, "Training is not reproducible!"

@pytest.mark.ml_test_score(category_test="Infra3", status="automatic")
# Infra 3: Integration test reproducibility of DVC pipeline
def test_dvc(threshold):
    """Test that the DVC pipeline runs and produces expected outputs"""
    # run the DVC pipeline
    result = subprocess.run(
                ['dvc', 'repro', "--force"],
                capture_output=True,
                text=True,
                check=False
                )
    assert result.returncode == 0, "DVC repro failed!"

    # check if files were created
    model_path = config.DEFAULT_CLASSIFIER_MODEL_PATH
    metrics_path = config.DEFAULT_METRICS_PATH
    assert model_path.exists(), "Model file was not created!"
    assert metrics_path.exists(), "Metrics file was not created!"

    # open metrics file and check accuracy
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    assert 'accuracy' in metrics, "Metrics file does not contain accuracy!"
    assert metrics['accuracy'] >= threshold, f"Model accuracy is below threshold {threshold}    !"

# test argument parser creation
def test_create_argument_parser():
    """Test that the argument parser is created correctly."""
    parser = predict.create_argument_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args([])
    assert hasattr(args, 'bow_p')
    assert hasattr(args, 'model_p')