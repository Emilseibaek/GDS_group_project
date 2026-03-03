import pandas as pd



chunksize = 10000
total_rows = 995000
processed_rows = 0

for chunks in pd.read_csv("cleaned_for_classification.csv", chunksize=chunksize, nrows=total_rows):
    
    #split data in 80%, 10%, 10% randomly
    traning_data = chunks.sample(frac=0.8)
    test_data = chunks.sample(frac=0.1)
    validation_data = chunks.sample(frac=0.1)

    #save data
    traning_data.to_csv("training_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    validation_data.to_csv("validation_data.csv", index=False)