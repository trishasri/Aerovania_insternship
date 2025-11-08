# Decision Tree Visualizer (WineMag Dataset)
This project trains a **Decision Tree Classifier** on the `winemag-data-130k-v2.csv` dataset to classify wines by country and exports a PNG image of the trained tree.

---

## 📂 Files
- `dt_visualizer.py` — Main Python script.
- `decision_tree.png` — Exported visualization of the trained decision tree.
- `decision_tree_model.joblib` — Saved Decision Tree model.
- `scaler.joblib` — Saved feature scaler.
- `requirements.txt` — Python dependencies list.

---

# Add Dataset
Place your winemag-data-130k-v2.csv file in the same folder as this script

# Notes
```
The script automatically keeps only the top 5 most common wine-producing countries to prevent errors from rare categories.
Target variable: country
Features used: points, price, province, variety
Model: DecisionTreeClassifier(max_depth=4, random_state=42)
Visualization generated using sklearn.tree.plot_tree.
```
# Run Script
```
python dt_visualizer.py
```
# Output
```
After successful execution, you’ll find the following files in the same folder:
decision_tree.png — visual representation of the tree.
decision_tree_model.joblib — saved model for reuse.
scaler.joblib — saved StandardScaler object.
```
