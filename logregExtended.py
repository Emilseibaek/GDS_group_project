import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix






#fitNaiveBayes(content,type)

def binarytype(label):
    if label!="reliable":
        return "fake"
    return "reliable"


extendedNews=pd.read_csv("extended_cleaned_for_classification.csv")

def logregExtended(df):
    
    df=df.dropna(subset=["cleaned_text"])
    df=df.dropna(subset=["title"])
    df=df.dropna(subset=["domain"])
    
    df["combind"]=df["cleaned_text"]+" "+df["title"]+" "+df["domain"]
    
    X=df["domain"]
    y=list(map(binarytype,df["type"]))
    
    X_train,X_test_val,y_train,y_test_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    X_test, X_val, y_test, y_val= train_test_split(X_test_val,y_test_val,test_size=0.5,random_state=42,stratify=y_test_val)

    vectorizer=CountVectorizer(max_features=10000)
    
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_val=vectorizer.transform(X_val)

    
    model = LogisticRegression(max_iter=1000,C=0.1,class_weight=None) 
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Validation F1:", f1_score(y_val, y_val_pred, average='binary', pos_label="fake"))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred, labels=["fake", "reliable"]))

    y_test_pred = model.predict(X_test)
    print("Test F1:", f1_score(y_test, y_test_pred, average="binary", pos_label="fake"))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred, labels=["fake", "reliable"]))

logregExtended(extendedNews)