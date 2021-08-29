import pandas as pd

file_dir = '../data/patent_en_ko.csv'
df = pd.read_csv(file_dir)
print(df.head(3))
