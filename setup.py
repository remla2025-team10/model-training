from setuptools import setup, find_packages

setup(
    name="restaurant_model_training",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
) 