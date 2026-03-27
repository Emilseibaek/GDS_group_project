

Place `995,000_rows.csv` in the project root directory. Install dependencies:

```bash
pip install -r requirements.txt
```

The LIAR dataset files (`train.tsv`, `test.tsv`, `valid.tsv`) must also be in the project root.


Step 1: Preprocess the raw data

```bash
python PreprocessingWClassification.py
```

**Input:** `995,000_rows.csv`
**Output:** `extended_cleaned_for_classification.csv` — cleaned text with columns: `type`, `domain`, `title`, `cleaned_text`

Step 2: Split into train/test/validation

```bash
python TextSpliter.py
```

**Input:** `cleaned_for_classification.csv`
**Output:**
- `training_data.csv` — 80% of the data
- `test_data.csv` — 10% of the data
- `validation_data.csv` — 10% of the data

Each file has columns: `type`, `domain`, `title`, `cleaned_text`


Step 3 Run the Simple Logistic Regression model

```bash
python Simplelogisticregression.py
```

**Input:** `training_data.csv`, `test_data.csv`, `validation_data.csv`, `test.tsv`, `valid.tsv`
**Output:** Prints F1 scores, accuracy, and confusion matrices for both FakeNewsCorpus and LIAR datasets.

---
step 4 Run the Advanced SVM model

```bash
python advancedlogisticregression.py
```

**Input:** `training_data.csv`, `test_data.csv`, `validation_data.csv`, `test.tsv`, `valid.tsv`
**Output:** Prints F1 scores, accuracy, and confusion matrices for both FakeNewsCorpus and LIAR datasets.

other scripts

```bash
python ExplorationPipeline.py
```

**Input:** `cleaned_words_995k.csv`
**Output:** Prints word frequency statistics (URL, date, number counts and top 100 words).

