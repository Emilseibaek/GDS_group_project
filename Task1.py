from cleantext import clean
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import nltk


df_lib = pd.read_csv("995,000_rows.csv")

text_cols = df_lib.select_dtypes(include='object').columns
for col in text_cols:
    df_lib[col] = df_lib[col].apply(lambda text: clean(text,
        fix_unicode=False,               # fix various unicode errors
        to_ascii=False,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_code=False,                  # replace all code snippets with a special token
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_ip_addresses=False,          # replace all IP addresses with a special token
        no_file_paths=False,            # replace all file paths with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        exceptions=None,                # list of regex patterns to preserve verbatim
        replace_with_code="<CODE>",
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_ip_address="<IP>",
        replace_with_file_path="<FILE_PATH>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    ))



processed_words = ' '.join(df_lib.astype(str).values.flatten())

tokens = nltk.word_tokenize(processed_words)


unique_words_aftertokenization = set(tokens)

stop_words = set(stopwords.words('english'))

filtered_tokens = [word for word in tokens if word not in stop_words]

unique_words_afterstopwordremoval = set(filtered_tokens)

stemmer = PorterStemmer()

stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

unique_words_afterstemming = set(stemmed_tokens)



print(tokens[:1000])
print(filtered_tokens[:1000])
print(stemmed_tokens[:1000])    
print(len(unique_words_aftertokenization))
print(len(unique_words_afterstopwordremoval))
print(len(unique_words_afterstemming))

df = pd.DataFrame({'Wordsafterstemming': unique_words_afterstemming})
df.to_csv('wordsafterstemming.csv', index=False)
