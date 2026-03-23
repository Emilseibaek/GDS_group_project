import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,make_scorer,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import GridSearchCV





Liar_train_df = pd.read_csv("train.tsv", sep='\t', header=None).dropna(subset=[2])
Liar_test_df  = pd.read_csv("test.tsv",  sep='\t', header=None).dropna(subset=[2])
Liar_val_df   = pd.read_csv("valid.tsv", sep='\t', header=None).dropna(subset=[2])

X_train = Liar_train_df.iloc[:, 2]
X_test  = Liar_test_df.iloc[:, 2]
X_val   = Liar_val_df.iloc[:, 2]

y_train = Liar_train_df.iloc[:, 1]
y_test  = Liar_test_df.iloc[:, 1]
y_val   = Liar_val_df.iloc[:, 1]


# print("Train class distribution:", pd.Series(y_train).value_counts())
# print("Test class distribution:",  pd.Series(y_test).value_counts())
# print("Val class distribution:",   pd.Series(y_val).value_counts())




vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)
X_val   = vectorizer.transform(X_val)


model=LinearSVC(C=0.1,dual=True,max_iter=1000,penalty='l2')


model.fit(X_train,y_train)

y_val_pred=model.predict(X_val)

print("Validation F1:", f1_score(y_val, y_val_pred, average='macro'))
print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred,
      labels=['false','half-true','mostly-true','true','barely-true','pants-fire']))

y_test_pred=model.predict(X_test)

print("Test F1:", f1_score(y_test, y_test_pred, average='macro'))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred,
      labels=['false','half-true','mostly-true','true','barely-true','pants-fire']))


# y_test_pred = model.predict(X_test)
# print("Test F1:", f1_score(y_test, y_test_pred, average='macro'))
# print("Test Confusion Matrix:")
# print(confusion_matrix(y_test, y_test_pred, labels=['false', 'half-true', 'mostly-true', 'true', 'barely-true', 'pants-fire']))



# param_grid = {'dual':[True,False],'penalty':['l1','l2'],'max_iter':[1000,2000]}
# scorer = make_scorer(f1_score, average='binary', pos_label="fake")

# grid = GridSearchCV(
#     LinearSVC(C=0.1, class_weight='balanced'),
#     param_grid,
#     scoring=scorer,
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

# grid.fit(X_train, y_train)
# print("Best C:", grid.best_params_)
# print("Best F1:", grid.best_score_)
