import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def binarytype(label):
    if label!="reliable":
        return "fake"
    return "reliable"

train_df = pd.read_csv("training_data.csv").dropna(subset=["cleaned_text"])
test_df  = pd.read_csv("test_data.csv").dropna(subset=["cleaned_text"])
val_df   = pd.read_csv("validation_data.csv").dropna(subset=["cleaned_text"])

X_train = train_df["cleaned_text"]
X_test  = test_df["cleaned_text"]
X_val   = val_df["cleaned_text"]

y_train = list(map(binarytype, train_df["type"]))
y_test  = list(map(binarytype, test_df["type"]))
y_val   = list(map(binarytype, val_df["type"]))

vectorizer = CountVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)
X_val   = vectorizer.transform(X_val)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation F1:", f1_score(y_val, y_val_pred, average='binary', pos_label="fake"))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred, labels=["fake", "reliable"]))

y_test_pred = model.predict(X_test)
print("Test F1:", f1_score(y_test, y_test_pred, average="binary", pos_label="fake"))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=["fake", "reliable"]))