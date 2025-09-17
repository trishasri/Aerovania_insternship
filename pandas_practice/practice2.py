import pandas as pd
df = pd.read_csv("student.csv")
print(df)

#info ans stats
print(df.info())
print(df.describe())
print(df.dtypes)

#first and last rows
print(df.head())
print(df.tail())

#col selection
print(df.iloc[[1,34,52]])
print(df["Age"])
print(df.loc[2 , "StudentID"])
print(df.loc[2])

#filter data
print(df[df["Age"]>15])
print(df[df["Gender"]== 1])

#central tendencies
print(df["Age"].mean())
print(df["Age"].mode())
print(df["Age"].median())

#grouping
print(df.groupby("Age")["GPA"].mean())

#sort
print(df.sort_values("Age", ascending = True))

#missing values
print(df.isnull().sum())
# no missing values here but if present
#handle them 
print(df.fillna("GPA").mean())
#remove missing values
print(df.dropna())