import pandas as pd

df = []

for i in range(1000):
    print(i)
    df.append(pd.read_csv("/Users/ryanyohnk/Downloads/orig.csv"))

print(pd.concat(df).to_csv("/Users/ryanyohnk/Downloads/data.csv"))