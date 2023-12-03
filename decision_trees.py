import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

with open('best_selected_features_dectrees_rfe_grid.txt', 'r') as file:
    selected_features = [line.strip() for line in file.readlines()]

X = df_encoded[selected_features]
y = df_encoded['funding_total_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30, 50, 100],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', None]
}

dt = DecisionTreeRegressor(random_state=42)

dt_grid = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, 
                       verbose=2, n_jobs=-1)

dt_grid.fit(X_train, y_train)

best_params = dt_grid.best_params_
print("Best Hyperparameters:")
print(best_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
rmse_scores = []
r2_scores = []
medae_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_kf, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    best_model = DecisionTreeRegressor(**best_params, random_state=42)

    best_model.fit(X_train_kf, y_train_kf)

    y_pred_val = best_model.predict(X_val)

    mse_scores.append(mean_squared_error(y_val, y_pred_val))
    mae_scores.append(mean_absolute_error(y_val, y_pred_val))
    rmse_scores.append(mean_squared_error(y_val, y_pred_val, squared=False))
    r2_scores.append(r2_score(y_val, y_pred_val))
    medae_scores.append(median_absolute_error(y_val, y_pred_val))

avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)
avg_medae = np.mean(medae_scores)

final_model = DecisionTreeRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

y_pred_test = final_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
medae_test = median_absolute_error(y_test, y_pred_test)

print("Cross-Validation Metrics:")
print(f"Average Mean Squared Error (MSE) across folds: {avg_mse}")
print(f"Average Mean Absolute Error (MAE) across folds: {avg_mae}")
print(f"Average Root Mean Squared Error (RMSE) across folds: {avg_rmse}")
print(f"Average R-squared (R2) across folds: {avg_r2}")
print(f"Average Median Absolute Error (MedAE) across folds: {avg_medae}")

print("\nTest Set Metrics:")
print(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse_test}")
print(f"R-squared (R2) on Test Set: {r2_test}")
print(f"Median Absolute Error (MedAE) on Test Set: {medae_test}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, c='blue', label='Actual vs Predicted', marker='o')
plt.scatter(y_test, y_test, alpha=0.5, c='red', label='Perfect Prediction', marker='x') 
plt.title('Predicted vs Actual Values (Test Set)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
