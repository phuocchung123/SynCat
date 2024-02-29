from setuptools import setup, find_packages

setup(
    name='ReCatAI',
    version='0.1.0',
    description='Reaction Category Prediction using Reaction Fingerprint',
    author='Phuoc-Chung Van Nguyen',
    author_email='your.email@example.com',
    url='https://github.com/phuocchung123/reaction_classification',
    packages=find_packages(),
    install_requires=[
        'rdkit==2023.9.5',
        'scikit-learn==1.4.1',
        'xgboost==2.0.3',
        'catboost==1.2.3',
        'joblib==1.3.2'
        # add other dependencies as needed
    ],
    python_requires='==3.11',
    # Additional metadata like classifiers, keywords, etc.
)