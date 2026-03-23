import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def binarytype(label):
    if label != "reliable":
        return "fake"
    return "reliable"


# Load data
train_df = pd.read_csv("training_data.csv").dropna(subset=["cleaned_text"])
test_df  = pd.read_csv("test_data.csv").dropna(subset=["cleaned_text"])
val_df   = pd.read_csv("validation_data.csv").dropna(subset=["cleaned_text"])


X_train = train_df["cleaned_text"]
X_test  = test_df["cleaned_text"]
X_val   = val_df["cleaned_text"]

y_train = list(map(binarytype, train_df["type"]))
y_test  = list(map(binarytype, test_df["type"]))
y_val   = list(map(binarytype, val_df["type"]))


# Pipeline: vectorizer + model
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("model", LinearSVC(class_weight="balanced"))
])


# Hyperparameter grid
param_grid = {
    "vectorizer__ngram_range": [(1,1), (1,2)],
    "vectorizer__max_features": [10000, 20000],
    "vectorizer__min_df": [2, 5],
    "vectorizer__max_df": [0.9, 0.95],
    "vectorizer__stop_words": ["english"],
    "vectorizer__sublinear_tf": [True, False],
    "model__C": [0.01, 0.1, 1, 5]
}


# Scoring (F1 for fake class)
scorer = make_scorer(f1_score, average="binary", pos_label="fake")


# Grid search
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)


# Best model
print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_model = grid.best_estimator_


# Validation evaluation
val_pred = best_model.predict(X_val)

print("\nValidation F1:",
      f1_score(y_val, val_pred, average="binary", pos_label="fake"))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred, labels=["fake", "reliable"]))


# Test evaluation
test_pred = best_model.predict(X_test)

print("\nTest F1:",
      f1_score(y_test, test_pred, average="binary", pos_label="fake"))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=["fake", "reliable"]))

print("Test Accuracy:",
      accuracy_score(y_test, test_pred))