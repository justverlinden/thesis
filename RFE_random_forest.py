import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

X = df_encoded.drop(columns=['funding_total_usd'])
y = df_encoded['funding_total_usd']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
best_selected_features = None
best_feature_importances = None
best_score = float('-inf')
best_num_features = 0

for num_features_to_select in range(100, 301, 25):  
    print(f"Selecting {num_features_to_select} features...")
    rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)
    print("Fitting RFE...")
    rfe.fit(X_train, y_train)

    selected_features = X.columns[rfe.support_]
    print("Selected features:", selected_features)

    print("Fitting model...")
    model.fit(X_train[selected_features], y_train)
    score = model.score(X_val[selected_features], y_val)
    print(f"Score with {num_features_to_select} features:", score)

    if score > best_score:
        best_score = score
        best_num_features = num_features_to_select
        best_selected_features = selected_features

np.savetxt('best_selected_features_randomforest_rfe.txt', best_selected_features, fmt='%s')

with open('best_feature_importance_scores_rfe_sorted_randomforest.txt', 'w') as f:
    rfe = RFE(estimator=model, n_features_to_select=best_num_features)
    rfe.fit(X_train, y_train)
    sorted_features = sorted(zip(X.columns[rfe.support_], rfe.estimator_.feature_importances_), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        f.write(f"{feature}: {importance}\n")

print(f"Best number of features: {best_num_features}")
print(f"Best score: {best_score}")
print(f"Best selected features: {best_selected_features}")
