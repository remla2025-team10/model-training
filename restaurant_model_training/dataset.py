import pandas as pd
from preprocess_sentiment_analysis import preprocess_dataframe

def get_data(raw_data_path, processed_data_path):
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)
    corpus = preprocess_dataframe(dataset, output_path=str(processed_data_path))['Processed_Review']
    labels = dataset.loc[:, 'Liked'].values

    return corpus, labels
