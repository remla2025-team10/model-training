from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = PROJECT_ROOT / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_RAW_DATA_PATH = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
DEFAULT_RAW_DATA_FRESH_PATH = RAW_DATA_DIR / "a2_RestaurantReviews_FreshDump.tsv"
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_reviews.csv"
DEFAULT_BOW_MODEL_PATH = MODELS_DIR / "BoW_Sentiment_Model.pkl"
DEFAULT_CLASSIFIER_MODEL_PATH = MODELS_DIR / "Classifier_Sentiment_Model.joblib"
DEFAULT_METRICS_PATH = METRICS_DIR / "metrics.json"

DEFAULT_MAX_FEATURES = 1420

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 0