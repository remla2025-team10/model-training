# model-training

This repository trains a Restaurant Review Sentiment Analysis model using the Naive Bayes classifier.

# Project Setup

## 1. Clone the Repository

```bash
git clone https://github.com/remla2025-team10/model-training.git
cd model-training
```

## 2. Set Up Virtual Environment

### For Unix/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

Make sure you have `pip` up to date:

```bash
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## 4. Run the training script

### With default parameters:

```bash
python train.py
```

### With custom parameters:

```bash
python train.py \
  --data_p data_path_of_training_data.tsv \
  --processed_p save_path_for_processed_reviews.csv \
  --bow_p save_path_for_bow_vectorizer.pkl \
  --model_p save_path_for_trained_model \
  --bow_max_features 1000
```

## 5. Deactivate the Virtual Environment (When Done)

```bash
deactivate
```
## Remote Model Repository

This repository uses [Data Version Control (DVC)](https://dvc.org/) to manage machine learning datasets and model artifacts. We've configured a Google Drive remote storage to facilitate collaboration without duplicating large data files in Git.

### Dependencies

Make sure you have `DVC` and `dvc-gdrive` installed. If not, you can run `pip install -r requirements.txt` to install all dependencies, including DVC and dvc-gdrive.

### Remote Storage Configuration

We use Google Drive as our DVC remote storage. The configuration can be set with a script **(Please reach out to us to get the bash file with credentials)**, and it will been set up in the `.dvc/config` file. You can check the current remote configuration with:

```bash
dvc remote list
```

### Woring with DVC

After making changes to tracked data files, reproduce the pipeline:

```bash
dvc repro
```

Push your data changes to the remote:

```bash
dvc push
```

Pull the latest changes from the remote:

```bash
dvc pull
```


## Notes

- The code is tested on Python 3.12.