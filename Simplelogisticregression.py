import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix


def fakenews_binarytype(label):
    if label != "reliable":
        return "fake"
    return "reliable"

def Liar_binarytype(label):
    if label in ['false','barely-true','pants-fire']:
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



vectorizer = CountVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)
X_val   = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train, y_train)
#Fake news corpus

y_val_pred = model.predict(X_val)
print("Validation F1:", f1_score(y_val, y_val_pred, average='binary', pos_label="fake"))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred, labels=["fake", "reliable"]))

y_test_pred = model.predict(X_test)
print("Test F1:", f1_score(y_test, y_test_pred, average="binary", pos_label="fake"))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=["fake", "reliable"]))

print("accuracy",accuracy_score(y_test,y_test_pred))

#Liar Corpus

#Liar Val evaluation
Liar_X_val  = vectorizer.transform(Liar_X_val)
Liar_X_test = vectorizer.transform(Liar_X_test)

Liar_val_pred=model.predict(Liar_X_val)

print("Validation F1:", f1_score(Liar_y_val, Liar_val_pred,average='binary',pos_label='fake'))
print("Validation accuracy:", accuracy_score(Liar_y_val, Liar_val_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(Liar_y_val, Liar_val_pred,labels=["fake", "reliable"]))

#Liar Test Evaluation
Liar_test_pred=model.predict(Liar_X_test)

print("Test F1:", f1_score(Liar_y_test, Liar_test_pred, average='binary',pos_label='fake'))
print("Test accuracy:", accuracy_score(Liar_y_test, Liar_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(Liar_y_test, Liar_test_pred,labels=["fake", "reliable"]))

