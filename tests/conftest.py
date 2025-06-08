import pytest
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

    # define temp files
    tmp_path = tmp_path_factory.mktemp("model_test")
    data_p = str(RAW_DATA_PATH)
    processed_p = tmp_path / "processed.csv"
    model_p = tmp_path / "model.joblib"
    bow_p = tmp_path / "bow.pkl"

    # get data and features
    corpus, labels = get_data(data_p, processed_p)
    features = create_bow_features(corpus, max_features=config.DEFAULT_MAX_FEATURES, bow_path=bow_p)

    # train model
    train.train_model(features, labels, model_p, test_size=0.2, random_state=42)
    vectorizer, classifier = predict.load_models(bow_p, model_p)
    
    return vectorizer, classifier