import joblib
from restaurant_model_training.modeling import train
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from restaurant_model_training.dataset import get_data
from restaurant_model_training.features import create_bow_features
from restaurant_model_training import config

# Infra 1: test reproducibility of training process
def test_reproducibility(tmp_path):
    data_p = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_p1 = tmp_path / "proc1.csv"
    processed_p2 = tmp_path / "proc2.csv"
    model_p1 = tmp_path / "model1.joblib"
    model_p2 = tmp_path / "model2.joblib"
    bow_p1 = tmp_path / "bow1.pkl"
    bow_p2 = tmp_path / "bow2.pkl"

    # get data and features
    corpus1, labels1 = get_data(raw_data_path=data_p, processed_data_path=processed_p1)
    corpus2, labels2 = get_data(raw_data_path=data_p, processed_data_path=processed_p2)
    features1 = create_bow_features(corpus=corpus1, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p1)
    features2 = create_bow_features(corpus=corpus2, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p2)

    # run training twice with same seed and params
    train.train_model(features1, labels1, model_p1, test_size=0.2, random_state=100)
    train.train_model(features2, labels2, model_p2, test_size=0.2, random_state=100)

    model1 = joblib.load(model_p1)
    model2 = joblib.load(model_p2)

    # same data, same seed -> same accuracy
    df = pd.read_csv(data_p, delimiter="\t", quoting=3)
    X = pickle.load(open(bow_p1, 'rb')).transform(df['Review']).toarray()
    y = df['Liked']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    acc1 = accuracy_score(y_test, model1.predict(X_test))
    acc2 = accuracy_score(y_test, model2.predict(X_test))

    assert abs(acc1 - acc2) < 0.01, "Training is not reproducible!"
