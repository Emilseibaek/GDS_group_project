import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp






def binarytype(label):
    if label!="reliable":
        return "fake"
    return "reliable"

News=pd.read_csv("sample_cleaned_for_classification.csv")



def neural(df):
    df = df.dropna(subset=["cleaned_text"])

    X=df["cleaned_text"]
    y=list(map(binarytype,df["type"]))
    
    X_train_val,X_test,Y_train_val,Y_test=train_test_split(X,y,test_size=0.1,random_state=42,stratify=y)

    X_train, X_val, Y_train, Y_val= train_test_split(X_train_val,Y_train_val,test_size=0.1,random_state=42,stratify=Y_train_val)

    vectorizer=CountVectorizer(max_features=10000,min_df=5)

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_val=vectorizer.transform(X_val)

    #added ballanced weight remeber to test again.
    model = ComplementNB()
    model.fit(X_train, Y_train)

    y_val_pred = model.predict(X_val)
    print("Validation F1:",f1_score(Y_val, y_val_pred,average='binary',pos_label="fake"))
    y_test_pred = model.predict(X_test)
    #print("Test F1:", f1_score(Y_test,y_test_pred,average="binary",pos_label="fake"))

neural(News)