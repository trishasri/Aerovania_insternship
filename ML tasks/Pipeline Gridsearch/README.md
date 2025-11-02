# Pipeline + GridSearchCV (Titanic-like Dataset)

##  Overview
This project builds an end-to-end **machine learning pipeline** using `scikit-learn` to:
- Impute missing values  
- Encode categorical features  
- Scale numeric features  
- Train a **RandomForestClassifier**  
- Tune hyperparameters using **GridSearchCV**  

The model is trained and evaluated on a Titanic-like dataset (`train_and_test2.csv`), and the best model is saved using **joblib**.

---

### Files Included
PIPELINE_GRIDSEARCH:
- pipeline_gridsearch.ipynb - Main Jupyter notebook with preprocessing, pipeline, training, tuning, and evaluation 
- train_and_test2.csv - Input dataset (Titanic-like) 
- best_model.joblib -Serialized best-performing model
- grid_results.csv - CSV file containing all GridSearchCV results
- feature_importances.csv  (optional) - Feature importance values (if exported)
- README.md - Project documentation 
- requirements.txt -Required Python packages 
---

##  How to Run

###  Install dependencies
```bash
pip install -r requirements.txt

```
### Run the notebook
Open and execute:
```
 jupyter notebook pipeline_gridsearch.ipynb
```

## Output
After execution, you will get:
```
best_model.joblib - saved model
grid_results.csv - grid search cross-validation results
Accuracy, classification report, and confusion matrix printed in output
Optional feature importance plot
```
## Example Output
```
Best Parameters: {'clf__max_depth': 6, 'clf__min_samples_split': 2, 'clf__n_estimators': 200}
Accuracy: 0.83
Confusion Matrix:
[[95 12]
[18 58]]
```

## Notes:
- The target column in  dataset is 2urvived.
- The pipeline handles missing values, categorical encoding, and feature scaling automatically.
- At least 3 hyperparameters are tuned using GridSearchCV:
  - n_estimators
  - max_depth
  - min_samples_split


