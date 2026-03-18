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
total_rows = 90000
processed_rows = 0

stop_words = set()
for lang in ['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'turkish']:
    stop_words.update(stopwords.words(lang))
stemmer = PorterStemmer()

# Create or clear the output CSV
output_csv = 'extended_cleaned_for_classification.csv'
with open(output_csv, 'w') as f:
    f.write('type,domain,title,cleaned_text\n')

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

for chunk in pd.read_csv("995,000_rows.csv", chunksize=chunksize, nrows=total_rows, low_memory=False):
    
    rows = []  

    for _, row in chunk.iterrows():
        label = row.get("type", None)
        Dlabel=row.get("domain",None)
        Tlabel=row.get("title",None)

        # Skip rows without a label
        
        if pd.isna(label):
            continue

        text = str(row.get("content", ""))
        cleaned = cleantext(text)

        tokens = nltk.word_tokenize(cleaned)
        stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words and w.isalnum()]

        rows.append({
            "type": label,
            "Domain":Dlabel,
            "title":Tlabel,
            "cleaned_text": " ".join(stemmed)
        })

    chunk_df = pd.DataFrame(rows)
    chunk_df.to_csv(output_csv, mode='a', header=False, index=False)

    processed_rows += len(chunk)
    print(f"Processed {processed_rows}/{total_rows} rows...")

print(f"Done! Saved to {output_csv}.")
