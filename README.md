Fake News Project

Overview:
This project focuses on detecting fake news articles.
We preprocess a large-scale dataset and train classification models (Logistic Regression and Support Vector Machine) using TF-IDF representations.

The final model achieves an F1 score of aprox. 0.96, indicating strong performance on binary fake vs. reliable classification.

Dataset:

FakeNewsCorpus dataset (aprox. 995,000 articles).

Each article contains:
content (main text)
type (label)
Labels are grouped into:
reliable
fake (all other categories)


Installation
Install required libraries:
pip install pandas nltk scikit-learn clean-text
Download NLTK resources (automatically handled in scripts).

Pipeline Overview

The project follows this pipeline:
Raw dataset -> preprocessing -> cleaned dataset -> splitting -> model training -> evaluation

Project Structure

PreprocessingWClassification.py
Cleans text, removes noise, applies stemming, and saves cleaned dataset

ExplorationPipeline.py
Performs frequency analysis and token statistics

TextSpliter.py
Splits dataset into:
80% training
10% validation
10% test

svm.py
Trains and evaluates the SVM model using GridSearchCV

How to Run:
1. Preprocess data
python PreprocessingWClassification.py
2. Split dataset
python TextSpliter.py
3. Train and evaluate model
python svm.py

Models:
Logistic Regression
Baseline model
Simple TF-IDF representation
Support Vector Machine (SVM)
Uses LinearSVC
Optimized with GridSearchCV
Best hyperparameters:
C = 0.1
max_features = 10000
min_df = 5
ngram_range = (1,1)

Results

Model	Validation F1	Test F1	Accuracy
Logistic Regression	~0.96	~0.96	~0.93
SVM (best)	~0.96	~0.96	~0.94

Confusion matrix (SVM):

Few false positives and false negatives
Balanced performance across classes

Observations

Preprocessing significantly improves performance by:
Removing noise (URLs, numbers, dates)
Reducing vocabulary size
Word frequency follows Zipf’s law
Stopword removal mainly affects high-frequency words
