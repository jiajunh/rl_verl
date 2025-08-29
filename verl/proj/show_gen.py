import pandas as pd
import json

# Load the Parquet file
filename = "/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/aime2024_multiturn/test"

df = pd.read_parquet(filename + ".parquet")

print(df.shape)
print(df.columns.tolist())
idx = 100
for col in df.columns.tolist():
    print(df[col][idx])

