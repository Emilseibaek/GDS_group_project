import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def fakenews_binarytype(label):
    if label != "reliable":
        return "fake"
    return "reliable"

def Liar_binarytype(label):
    if label in['false','barely-true','pants-fire']:
        return "fake"
    return "reliable"


# Fake news data
train_df = pd.read_csv("training_data.csv").dropna(subset=["cleaned_text"])
test_df  = pd.read_csv("test_data.csv").dropna(subset=["cleaned_text"])
val_df   = pd.read_csv("validation_data.csv").dropna(subset=["cleaned_text"])

X_train = train_df["cleaned_text"]
X_test  = test_df["cleaned_text"]
X_val   = val_df["cleaned_text"]

y_train = list(map(fakenews_binarytype, train_df["type"]))
y_test  = list(map(fakenews_binarytype, test_df["type"]))
y_val   = list(map(fakenews_binarytype, val_df["type"]))

# Liar Data
Liar_test_df  = pd.read_csv("test.tsv",  sep='\t', header=None).dropna(subset=[2])
Liar_val_df   = pd.read_csv("valid.tsv", sep='\t', header=None).dropna(subset=[2])

Liar_X_test  = Liar_test_df.iloc[:, 2]
Liar_X_val   = Liar_val_df.iloc[:, 2]


Liar_y_test  = Liar_test_df.iloc[:,1].map(Liar_binarytype)
Liar_y_val   = Liar_val_df.iloc[:,1].map(Liar_binarytype)


# Pipeline: vectorizer + model
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1,2),min_df=5,max_df=0.9,sublinear_tf=True,max_features=20000)),
    ("model", LinearSVC(C=5))
])


pipeline.fit(X_train, y_train)


# Fake News Validation evaluation
val_pred = pipeline.predict(X_val)

print("\n Fake NewS Validation F1:",
      f1_score(y_val, val_pred, average="binary", pos_label="fake"))

print("Fake News validation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred, labels=["fake", "reliable"]))


# Fake News Test evaluation
test_pred = pipeline.predict(X_test)

print("\nFake News test F1:",
      f1_score(y_test, test_pred, average="binary", pos_label="fake"))

print("Fake News Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=["fake", "reliable"]))

print("Fake news Test Accuracy:",
      accuracy_score(y_test, test_pred))

#Liar Val evaluation

Liar_val_pred=pipeline.predict(Liar_X_val)

print("Validation F1:", f1_score(Liar_y_val, Liar_val_pred,average='binary',pos_label='fake'))
print("Validation accuracy:", accuracy_score(Liar_y_val, Liar_val_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(Liar_y_val, Liar_val_pred,labels=["fake", "reliable"]))

#Liar Test Evaluation
Liar_test_pred=pipeline.predict(Liar_X_test)

print("Test F1:", f1_score(Liar_y_test, Liar_test_pred, average='binary',pos_label='fake'))
print("Test accuracy:", accuracy_score(Liar_y_test, Liar_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(Liar_y_test, Liar_test_pred,labels=["fake", "reliable"]))

