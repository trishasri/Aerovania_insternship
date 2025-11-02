import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
# Load Dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\titanic.csv")
print("Initial shape:", df.shape)
# Handle Missing Values
# Fill numeric columns with median
num_imputer = SimpleImputer(strategy='median')
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill categorical columns with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Feature Engineering (Derived Features)
# Feature 1: Title from Name
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
else:
    df['Title'] = "Unknown"

# Feature 2: FamilySize = SibSp + Parch + 1
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 if 'SibSp' in df.columns and 'Parch' in df.columns else 1

# Feature 3: IsAlone (binary)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Feature 4: Age*Class interaction
if 'Age' in df.columns and 'Pclass' in df.columns:
    df['Age_Class'] = df['Age'] * df['Pclass']

# Feature 5: Fare per Person
if 'Fare' in df.columns:
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

# 4️⃣ Encode Categorical Variables
# Label encode binary categorical columns
label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() == 2:
        df[col] = label_enc.fit_transform(df[col])
    else:
        # One-hot encode multi-class categorical variables
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Save Processed Data
df.to_csv("processed_train.csv", index=False)
print("Feature engineering complete!")
print("Processed file saved as 'processed_train.csv'")
print("Final shape:", df.shape)