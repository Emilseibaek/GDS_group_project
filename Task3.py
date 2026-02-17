from cleantext import clean
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import nltk



df = pd.read_csv("wordsafterstemming.csv")

processed_words = ' '.join(df['Wordsafterstemming'].astype(str).values.flatten())

url_count = processed_words.lower().count('url')

date_count = processed_words.lower().count('date')

numeric_count = processed_words.lower().count('number')

most_common_words = Counter(processed_words.split()).most_common(100)

print(url_count)
print(date_count)
print(numeric_count)
print(most_common_words)
