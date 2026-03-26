import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import csv
import sys

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

nltk.download("punkt", quiet=True)

chunksize = 10000

#Before cleaning:
print("raw text:")

word_counter_raw = Counter()

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

#Top 10 000
top_raw = word_counter_raw.most_common(10000)
freqs_raw = [freq for _, freq in top_raw]

#After cleaning:
print("cleaned text:")

df_clean = pd.read_csv("cleaned_for_classification.csv")
texts_clean = df_clean["cleaned_text"].dropna().astype(str)

cleaned_words_string = " ".join(texts_clean)
cleaned_tokens = cleaned_words_string.split()

word_counter_clean = Counter(cleaned_tokens)

#Top 10 000
top_clean = word_counter_clean.most_common(10000)
freqs_clean = [freq for _, freq in top_clean]


#Plot:
plt.figure(figsize=(10, 6))

plt.plot(range(1, len(freqs_raw)+1), freqs_raw, label="Before Cleaning")
plt.plot(range(1, len(freqs_clean)+1), freqs_clean, label="After Cleaning")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Word Rank")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Top 10,000 Words)")
plt.legend()

plt.tight_layout()
plt.show()
