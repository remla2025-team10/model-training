# model-training

![Coverage](https://img.shields.io/badge/Coverage-94%25-brightgreen)










<!-- START_PYLINT_SCORE -->
![Pylint Score](https://img.shields.io/badge/Pylint%20Score-10.00%2F10-brightgreen)
<!-- END_PYLINT_SCORE -->











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

Before running the scripts, you need to ensure all of the large data files used for training the model are retrieved through DVC. Please check out [how this is done](#remote-model-repository) before proceeding with the training examples.

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

### Run the tests manually
```bash
pytest --cov=restaurant_model_training tests/

```

### Run pylint for code quality checks
```bash
pylint restaurant_model_training/ tests/
```

## 7. Deactivate the Virtual Environment (When Done)

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

### Adding our Google Drive Remote Storage
Run the following commands to install the secrets required for our Google Drive Remote Storage:

```bash
dvc remote modify --local myremote gdrive_client_id 375913846623-51tmacon66o5f53lqhro3f5kphoj1sgj.apps.googleusercontent.com
dvc remote modify --local myremote gdrive_client_secret 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
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

To run an experiment, execute `dvc exp run`, to compare and show metrics from different experiments, execute `dvc exp show`.



# Structure
The structure follows the established [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science) for data science projects. Some of the structure still contains empty files/folders, as they were created according to the template and may be used in the future.

The directories you should pay attention to are the following:
* `.dvc/config`: DVC config file
* `.github/`
    * `metric_reporting/` 
        * `ml_test_score.py`: Update README.md with ML test score
        * `performance_metrics.py`: Update README with performance metrics
        * `pylint_score.py`: Update README.md with pylint score
    * `workflows/`
        * `feature_bump.yml`: Pre-release versioning for feature branches
        * `tests.yml`: Code quality and tests
* `data/`: The folder containing all the data files
    * `raw/`: Original raw data dumps
    * `processed/`: The processed data directly used by the model
* `models/`: Containes the models which have already been trained
* `pylint_plugins/`: Directory for the pylint plugins for ML code smells
* `restaurant_model_training/`: The main package (module) of this project
    * `modeling/`: Module containing logic for model training(`train.py`) and predicting (`predict.py`)
    * `config.py`: Contains the configurations such as default values and paths
    * `dataset.py`: Logic for loading and preprocessing the data
    * `features.py`: Logic for creating BOW features
    * `get_data.py`: Loads the raw datasets for the restaurant sentiment prediction.
    * `preprocessing.py`: Preprocessing script (using `lib-ml` package)
* `tests/`: The test files for the model
    * `conftest.py`: Tests configuration for the restaurant model training package
    * `test_data_features.py`: Tests for data and features
    * `test_infrastructure.py`: Tests for infrastructure (e.g. DVC pipeline integration test)
    * `test_model_development.py`: Tests for model training, evaluation, robustness
    * `test_monitoring.py`: Tests for model monitoring (e.g. regression)
    * `test_mutamorphic.py`: Tests for mutamorphic equivalence
    * `test_non_functional.py`: Tests for non functional features (e.g. cost of memory and latency in feature generation)
* `.flake8`: Flake8 config file
* `bandit.yaml`: Bandit config file
* `dvc.yaml`: DVC pipeline
* `requirements.txt`: The project dependencies

## Notes

- The code is tested on Python 3.12.












<!-- START_PERFORMANCE_METRICS -->
## Performance Metrics

- **Accuracy**: 0.75
- **Precision**: 0.75
- **Recall**: 0.75
- **F1 Score**: 0.75

- **Confusion Matrix**: [[53, 29], [16, 82]]

<!-- END_PERFORMANCE_METRICS -->




















<!-- START_ML_TEST_SCORE -->
## ML Test Scores

**Features and Data**: 2

**Model Development**: 2

**ML Infrastructure**: 2

**Monitoring**: 3

### Final ML Test Score: 2

- There’s been first pass at basic productionization, but additional investment may be needed.

<!-- END_ML_TEST_SCORE -->









