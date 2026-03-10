import re
import pandas as pd
import nltk
from cleantext import clean
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

print("Reading and processing data in chunks...")

chunksize = 10000
total_rows = 995000
processed_rows = 0

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Create or clear the target CSV file and write the header
output_csv = 'cleaned_words_995k.csv'
with open(output_csv, 'w') as f:
    f.write('Cleaned_words_995k\n')

# Setup date pattern regex once
date_pattern = r'\b(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b'

def cleantext(text):
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

for chunk in pd.read_csv("995,000_rows.csv", low_memory=False, chunksize=chunksize, nrows=total_rows):
    
    
    chunk_stemmed_tokens = []
    text_cols = chunk.select_dtypes(include='object').columns
    
    for col in text_cols:
        valid_text = chunk[col].dropna().astype(str)
        cleaned_series = valid_text.apply(cleantext)
        
        for text in cleaned_series:
            raw_tokens = nltk.word_tokenize(text)
        
            for word in raw_tokens:
                if word not in stop_words and word.isalnum():
                    chunk_stemmed_tokens.append(stemmer.stem(word))
    
    
    chunk_df = pd.DataFrame(chunk_stemmed_tokens, columns=['Cleaned_words_995k'])
    chunk_df.to_csv(output_csv, mode='a', header=False, index=False)
    
    processed_rows += len(chunk)
    print(f"Processed {processed_rows}/{total_rows} rows...")

print(f"Task1 complete. All stemmed words export to {output_csv}.")
