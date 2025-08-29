import pandas as pd
import json

# Load the Parquet file
filename = "/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/deepseek/20.jsonl"


max_len = 0
idx = 0

with open(filename, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        print(type(data), len(data))
        # length = len(data["output"])
        # if length > max_len:
        #     max_len = max(max_len, length)
        #     idx = i
        #     print(data)

# print(f"max length: {max_len}")
