import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split






#fitNaiveBayes(content,type)

def binarytype(label):
    if label!="reliable":
        return "fake"
    return "reliable"

News=pd.read_csv("sample_cleaned_for_classification.csv",nrows=1000)



def linreg(df):
    
    df = df.dropna(subset=["cleaned_text"])

    X=df["cleaned_text"]
    y=list(map(binarytype,df["type"]))
    
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    vectorizer=CountVectorizer(max_features=10000)
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred,average='binary',pos_label="fake"))

linreg(News)