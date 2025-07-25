"""
This script preprocesses the raw restaurant reviews dataset.
"""
from pathlib import Path
import pandas as pd
from preprocess_sentiment_analysis import preprocess_dataframe
import config

def preprocess_data(raw_data_path, processed_data_path):
    """
    Takes the path to the raw dataset and the path to save the processed data to.
    Returns the preprocessed data and labels.
    """
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)
    corpus = preprocess_dataframe(dataset, output_path=str(processed_data_path))['Processed_Review']
    labels = dataset.loc[:, 'Liked'].values

    return corpus, labels

if __name__ == "__main__":
    # Be careful of the relative paths here, DVC executes this file from
    # the top directory containing .dvc
    Path(config.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    print('\n################### Processing ###################\n')
    preprocess_data(config.DEFAULT_RAW_DATA_PATH, config.DEFAULT_PROCESSED_DATA_PATH)