import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df =pd.read_pickle('train_dev.pkl')
print(df.columns)
print(df.head(5))
