import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,classification_report,accuracy_score
from sklearn.metrics import confusion_matrix




#LIAR
Liar_train_df = pd.read_csv("train.tsv", sep='\t', header=None).dropna(subset=[2])
Liar_test_df  = pd.read_csv("test.tsv",  sep='\t', header=None).dropna(subset=[2])
Liar_val_df   = pd.read_csv("valid.tsv", sep='\t', header=None).dropna(subset=[2])

X_train = Liar_train_df.iloc[:, 2]
X_test  = Liar_test_df.iloc[:, 2]
X_val   = Liar_val_df.iloc[:, 2]

y_train = Liar_train_df.iloc[:, 1]
y_test  = Liar_test_df.iloc[:, 1]
y_val   = Liar_val_df.iloc[:, 1]

print("Labels:", Liar_train_df.iloc[:, 1].unique())
print("Sample statements:", Liar_train_df.iloc[:, 2].head())
print(pd.Series(y_train).value_counts())

vectorizer = CountVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)
X_val   = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print("Validation F1:", f1_score(y_val, y_val_pred,average='macro'))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred, labels=['false', 'half-true', 'mostly-true', 'true', 'barely-true', 'pants-fire']))

print(classification_report(y_val, y_val_pred))


y_test_pred = model.predict(X_test)
print("Test F1:", f1_score(y_test, y_test_pred,average='macro'))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=['false', 'half-true', 'mostly-true', 'true', 'barely-true', 'pants-fire']))
print("accuracy:",accuracy_score(y_test,y_test_pred))

