import pandas as pd
titles = pd.DataFrame.from_csv("../sheets/Processed/Titles.csv", header=0)

titles.drop("titleType", axis=1, inplace=True)
print(titles.head(2))
pd.DataFrame.to_csv(titles, "../sheets/Processed/Titles.csv")



