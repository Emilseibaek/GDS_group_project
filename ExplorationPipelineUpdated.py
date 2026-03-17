import csv
import sys
import pandas as pd
import nltk
from collections import Counter

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

nltk.download("punkt", quiet=True)

chunksize = 10000

#before cleaning:
word_counter_raw = Counter()
print("Raw text:")

for chunk in pd.read_csv(
    "995,000_rows.csv",
    chunksize=chunksize,
    engine="python",
    on_bad_lines="skip"
):
    texts = chunk["content"].dropna().astype(str)

    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        word_counter_raw.update(tokens)

print("\nTop 100 words before cleaning:")
print(word_counter_raw.most_common(100))

#after cleaning
print("\nProcessing cleaned text...")

df_clean = pd.read_csv("cleaned_for_classification.csv")
texts_clean = df_clean["cleaned_text"].dropna().astype(str)

cleaned_words_string = " ".join(texts_clean)
cleaned_tokens = cleaned_words_string.split()

clean_word_counter = Counter(cleaned_tokens)

print("\nCleaned text results:")
print("URL count (cleaned):", clean_word_counter["url"])
print("DATE count (cleaned):", clean_word_counter["date"])
print("NUMBER count (cleaned):", clean_word_counter["number"])

print("\nTop 100 words after cleaning:")
print(clean_word_counter.most_common(100))
