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

## Notes

- The code is tested on Python 3.12.