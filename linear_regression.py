import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

with open('best_selected_features_forward_selection.txt', 'r') as file:
    selected_features = [line.strip() for line in file.readlines()]

X = df_encoded[selected_features]
y = df_encoded['funding_total_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
rmse_scores = []
r2_scores = []
medae_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    linear_reg_selected = LinearRegression()

    linear_reg_selected.fit(X_train_fold, y_train_fold)

    y_pred_fold = linear_reg_selected.predict(X_val_fold)

    mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))
    mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
    rmse_scores.append(mean_squared_error(y_val_fold, y_pred_fold, squared=False))
    r2_scores.append(r2_score(y_val_fold, y_pred_fold))
    medae_scores.append(median_absolute_error(y_val_fold, y_pred_fold))

avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)
avg_medae = np.mean(medae_scores)

print(f"Average Mean Squared Error (MSE) across folds: {avg_mse}")
print(f"Average Mean Absolute Error (MAE) across folds: {avg_mae}")
print(f"Average Root Mean Squared Error (RMSE) across folds: {avg_rmse}")
print(f"Average R-squared (R2) across folds: {avg_r2}")
print(f"Average Median Absolute Error (MedAE) across folds: {avg_medae}")

linear_reg_selected = LinearRegression()
linear_reg_selected.fit(X_train, y_train)
y_pred_test = linear_reg_selected.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
medae_test = median_absolute_error(y_test, y_pred_test)

print("\nMetrics on Test Set:")
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