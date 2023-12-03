import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

X = df_encoded.drop(columns=['funding_total_usd'])
y = df_encoded['funding_total_usd']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

best_score = float('-inf')
best_selected_features = None
best_hyperparameters = None
score_feature_count = []

for num_features_to_select in range(1, 101, 1):
    print(f"Selecting {num_features_to_select} features...")
    rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)
    print("Fitting RFE...")
    rfe.fit(X_train, y_train)

    selected_features = X.columns[rfe.support_]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train[selected_features], y_train)

    best_params = grid_search.best_params_

    print("Fitting model...")
    model.set_params(**best_params)
    model.fit(X_train[selected_features], y_train)
    score = model.score(X_val[selected_features], y_val)
    score_feature_count.append((score, len(selected_features)))
    print(f"Score with {len(selected_features)} features:", score)

    if score > best_score:
        best_score = score
        best_selected_features = selected_features
        best_hyperparameters = best_params

print("Scores with corresponding number of features:")
for score, feature_count in score_feature_count:
    print(f"Number of features: {feature_count}, Score: {score}")

np.savetxt('best_selected_features_dectrees_rfe_grid.txt', best_selected_features, fmt='%s')

with open('best_hyperparameters_dectrees_rfe_grid.txt', 'w') as f:
    f.write(str(best_hyperparameters))

print("Getting feature importance for the best selected features")
model.set_params(**best_hyperparameters)
model.fit(X_train[best_selected_features], y_train)
feature_importances = model.feature_importances_

with open('feature_importance_scores_best_dectrees_rfe_grid.txt', 'w') as f:
    sorted_features = sorted(zip(best_selected_features, feature_importances), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_features:
        f.write(f"{feature}: {importance}\n")

print("Best score obtained:", best_score)
print("Number of features selected:", len(best_selected_features))
