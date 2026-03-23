import pandas as pd

chunksize = 10000
total_rows = 995000
processed_rows = 0

# Clear and write headers before the loop so re-running doesn't duplicate data
for fname in ["training_data.csv", "test_data.csv", "validation_data.csv"]:
    with open(fname, 'w') as f:
        f.write('type,domain,title,cleaned_text\n')

for chunks in pd.read_csv("extended_cleaned_for_classification.csv", chunksize=chunksize, nrows=total_rows):
    
    chunks = chunks.sample(frac=1, random_state=42).reset_index(drop=True)
    
    #split data chunk in 80%, 10%, 10% randomly without overlapping
    traning_data = chunks.iloc[:int(len(chunks)*0.8)]
    test_data = chunks.iloc[int(len(chunks)*0.8):int(len(chunks)*0.9)]
    validation_data = chunks.iloc[int(len(chunks)*0.9):]

    #append data
    traning_data.to_csv("training_data.csv", mode='a', header=False, index=False)
    test_data.to_csv("test_data.csv", mode='a', header=False, index=False)
    validation_data.to_csv("validation_data.csv", mode='a', header=False, index=False)
    
    processed_rows += len(chunks)
    print(f"Processed {processed_rows}/{total_rows} rows...")

print("Done!")