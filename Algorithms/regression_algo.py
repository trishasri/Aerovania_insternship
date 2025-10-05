from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_squared_error, r2_score

#regression moels
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#load datatset
dataset = fetch_california_housing()
X, y = dataset.data, dataset.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (for SVR and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_lr))
print("  R²:", r2_score(y_test, y_pred_lr), "\n")

# Ridge Regression 
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_ridge))
print("  R²:", r2_score(y_test, y_pred_ridge), "\n")

#Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_lasso))
print("  R²:", r2_score(y_test, y_pred_lasso), "\n")

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_dt))
print("  R²:", r2_score(y_test, y_pred_dt), "\n")

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_rf))
print("  R²:", r2_score(y_test, y_pred_rf), "\n")

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
print("Gradient Boosting Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_gbr))
print("  R²:", r2_score(y_test, y_pred_gbr), "\n")

# Support Vector Regressor
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
print("Support Vector Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_svr))
print("  R²:", r2_score(y_test, y_pred_svr), "\n")

#K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("K-Nearest Neighbors Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_knn))
print("  R²:", r2_score(y_test, y_pred_knn), "\n")






