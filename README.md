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
python -m restaurant_model_training.modeling.train
```

### With custom parameters:

```bash
python -m restaurant_model_training.modeling.train \
    --data_p <path/to/raw_data.tsv> \
    --processed_p <path/to/processed_data.csv> \
    --bow_p <path/to/bow_model.pkl> \
    --model_p <path/to/classifier> \
    --bow_max_features 1000 \
    --test_size 0.15 \
    --random_state 10
```

## 5. (Optionally) Try out the prediction

### With default parameters:
```bash
python -m restaurant_model_training.modeling.predict
```

### With custom parameters
```bash
python -m restaurant_model_training.modeling.predict \
    --bow_p <path/to/bow_model.pkl> \
    --model_p <path/to/classifier> \
```

## 6. (Optionally) test the model

### Install local package (restaurant_model_training)
```bash
pip install -e .
```

### Run the tests manually
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

## 7. Deactivate the Virtual Environment (When Done)

```bash
deactivate
```

# Structure
The structure follows the established [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science) for data science projects. Some of the structure still contains empty files/folders, as they were created according to the template and may be used in the future.

The directories you should pay attention to are the following:
* `data/`: The folder containing all the data files
    * `raw/`: Original raw data dumps
    * `processed/`: The processed data directly used by the model
* `models/`: Containes the models which have already been trained
* `restaurant_model_training/`: The main package (module) of this project
    * `config.py`: Contains the configurations such as default values and paths
    * `dataset.py`: Logic for loading and preprocessing the data
    * `features.py`: Logic for creating BOW features
    * `modeling/`: Module containing logic for model training (`train.py`) and predicting (`predict.py`)
* `tests/`: The test files for the model  
    * `test_data_features.py`: Tests for data and features
    * `test_infrastructure.py`: Tests for infrastructure
    * `test_model_development.py`: Tests for model training, evaluation, robustness
    * `test_monitoring.py`: Tests for model monitoring
* `requirements.txt`: The project dependencies
* `setup.py`: The package setup file

## Notes

- The code is tested on Python 3.12.