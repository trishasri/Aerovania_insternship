# Feature Engineering Script (Titanic Dataset)

## Overview
This project demonstrates a feature engineering pipeline on the Titanic dataset.  
It cleans data, handles missing values, encodes categorical features, and creates 5 derived features.

# File structure
```
Feature_engineering
  - feature_engineering.py# Main feature engineering script
  - processed_train.csv# Cleaned and processed dataset (output)
  - README.md# Project documentation
  - Requirements.txt# Python dependencies
```
## How to run 
### Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```
### Execute the Script
```
python feature_engineering.py'
```
### Output
```
Processed file saved as **processed_train.csv**
```
## key steps
1. Handle Missing Values:
 - Numeric columns: Replaced with the median.
 - Categorical columns: Replaced with the most frequent value (mode).
2. Encode Categorical Variables:
 - Binary categories: Encoded using Label Encoding.
 - Multi-class categories: Encoded using One-Hot Encoding.
3. Create Derived Features:
```
Feature  	      Description
Title	          Extracted from the passenger’s name (e.g., Mr, Miss, Mrs).
FamilySize	      Total family members aboard = SibSp + Parch + 1.
IsAlone	          Indicates if a passenger was alone (binary 0/1).
Age_Class	      Interaction feature = Age × Pclass.
Fare_Per_Person	  Fare adjusted by family size.
```
## Example Feature Engineering
```
Original Feature 	Engineered Feature     	Description
-----------------------------------------------------------------------------------------
Name	             Title	            Extracted honorifics (Mr, Miss, Mrs, etc.)
SibSp, Parch	     FamilySize	        Combined family information
FamilySize	         IsAlone         	Flag to indicate solo travelers
Age, Pclass          Age_Class	        Interaction between socioeconomic status and age
Fare, FamilySize	 Fare_Per_Person	Adjusted fare for shared tickets

```