import re
import json
import pandas as pd
import nltk
from cleantext import clean
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Read a subset of data for exploration
print("Reading data...")
# Read data with pandas. Why Pandas? It provides efficient, vectorized operations for tabular data,
# handles missing values gracefully, and easily fits in memory for the sampled chunks (like 10,000 rows).
df_lib = pd.read_csv("995,000_rows.csv", low_memory=False, nrows=995000)

# Check for inherent problems in the raw data
print(df_lib.shape)
print(df_lib.isnull().sum())

# Focus on text columns
text_cols = df_lib.select_dtypes(include='object').columns

all_cleaned_tokens = []

for col in text_cols:
    # Drop NAs for text processing
    valid_text = df_lib[col].dropna().astype(str)
    
    # Apply cleaning
    print(f"Cleaning column: {col}...")
    
    def cleantext(text):
        date_pattern = r'\b(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b'
        text_no_dates = re.sub(date_pattern, ' DATE ', text)
        
        cleaned = clean(text_no_dates,
            fix_unicode=True,               
            to_ascii=True,                  
            lower=True,                     
            no_line_breaks=True,           
            no_urls=True,                  
            no_emails=True,                
            no_numbers=True,               
            no_punct=False,                 
            replace_with_url=" URL ",
            replace_with_email=" EMAIL ",
            replace_with_number=" NUMBER ",
            lang="en"                       
        )
        return cleaned

    cleaned_series = valid_text.apply(cleantext)
    
    for text in cleaned_series:
        all_cleaned_tokens.extend(nltk.word_tokenize(text))

print("Tokenization complete.")

# 1. Frequency calculation after basic tokenization (and cleaning)
tokens_freq = Counter(all_cleaned_tokens)

# 2. Stopword removal
print("Removing stopwords...")
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in all_cleaned_tokens if word not in stop_words and word.isalnum()]
filtered_freq = Counter(filtered_tokens)

# 3. Stemming
print("Stemming...")
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
stemmed_freq = Counter(stemmed_tokens)


stemmed_df = pd.DataFrame(stemmed_tokens, columns=['Cleaned_words_995k'])
stemmed_df.to_csv('cleaned_words_995k.csv', index=False)


