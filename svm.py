import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.pipeline import Pipeline


def binarytype(label):
    if label != "reliable":
        return "fake"
    return "reliable"


# Load data
train_df = pd.read_csv("training_data.csv",nrows=100000).dropna(subset=["cleaned_text"])
test_df  = pd.read_csv("test_data.csv",nrows=10000).dropna(subset=["cleaned_text"])
val_df   = pd.read_csv("validation_data.csv",nrows=10000).dropna(subset=["cleaned_text"])


X_train = train_df["cleaned_text"]
X_test  = test_df["cleaned_text"]
X_val   = val_df["cleaned_text"]

y_train = list(map(binarytype, train_df["type"]))
y_test  = list(map(binarytype, test_df["type"]))
y_val   = list(map(binarytype, val_df["type"]))


# Pipeline: vectorizer + model
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1,2),min_df=5,max_df=0.9,sublinear_tf=True,max_features=None)),
    ("model", LinearSVC(C=5))
])


# param_grid = {
#     'vectorizer__max_features': (1000, 2000, None),
#     'vectorizer__sublinear_tf': (True, False)
# }
# Best parameters: {'vectorizer__ngram_range': (1, 2), 'vectorizer__min_df': 5, 'vectorizer__max_df': 0.9, 'model__C': 5}
# Best CV F1: 0.9701993085162154


# Scoring (F1 for fake class)
# scorer = make_scorer(f1_score, average="binary", pos_label="fake")


# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     scoring=scorer,
#     cv=3,
#     n_jobs=1,
#     verbose=2
# )

pipeline.fit(X_train, y_train)


# Best model
# print("Best parameters:", grid.best_params_)
# print("Best CV F1:", grid.best_score_)

# best_model = grid.best_estimator_


# Validation evaluation
val_pred = pipeline.predict(X_val)

print("\nValidation F1:",
      f1_score(y_val, val_pred, average="binary", pos_label="fake"))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred, labels=["fake", "reliable"]))


# Test evaluation
test_pred = pipeline.predict(X_test)

print("\nTest F1:",
      f1_score(y_test, test_pred, average="binary", pos_label="fake"))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=["fake", "reliable"]))

print("Test Accuracy:",
      accuracy_score(y_test, test_pred))