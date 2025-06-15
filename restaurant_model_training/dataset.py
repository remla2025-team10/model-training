""""
This module reads the raw dataset, preprocesses it and returns preprocessed data and labels.
"""
import pandas as pd
from preprocess_sentiment_analysis import preprocess_dataframe

def get_data(raw_data_path, processed_data_path):
    """
    Takes the path to the raw dataset and the path to save the processed data to.
    Returns the preprocessed data and labels.
    """
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)
    corpus = preprocess_dataframe(dataset, output_path=str(processed_data_path))['Processed_Review']
    labels = dataset.loc[:, 'Liked'].values

    return corpus, labels
