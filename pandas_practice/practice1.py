import pandas as pd
data = {
    "Name": ["Trisha","Sri","taara"],
    "Section": ["A","B","C"],
    "Marks": ["10","20","30"],
}
df = pd.DataFrame(data)
print(df)

# Info and Status
print(df.info())
print(df.describe())

#indexing
df = pd.DataFrame(data, index = ['std1','std2','std3'])
print(df)

#select col and rows
print(df.loc["std1","Name"])
print(df.iloc[[2,1]])