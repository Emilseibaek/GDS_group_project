from cleantext import clean
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import nltk


# read cleaned words
df = pd.read_csv("cleaned_words_995k.csv")

# join all cleaned words
processed_words = ' '.join(df['Cleaned_words_995k'].astype(str).values.flatten())

# count url, date, number
url_count = processed_words.lower().count('url')

date_count = processed_words.lower().count('date')

numeric_count = processed_words.lower().count('number')

# count most common words
most_common_words = Counter(processed_words.split()).most_common(100)

# sort most common words
sort_most_common_words = sorted(most_common_words, key=lambda x: x[1], reverse=True)

print(url_count)
print(date_count)
print(numeric_count)
print(sort_most_common_words)


