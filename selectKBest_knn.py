import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression  
from sklearn.neighbors import KNeighborsRegressor

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

X = df_encoded.drop(columns=['funding_total_usd'])
y = df_encoded['funding_total_usd']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor()

best_score = float('-inf')
best_selected_features = None

for num_features_to_select in range(1, 602, 1):
    print(f"Selecting {num_features_to_select} features...")
    
    selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X.columns[selector.get_support()]

    print("Selected features:")
    print(", ".join(selected_features))

    print("Fitting model...")
    model.fit(X_train_selected, y_train)
    X_val_selected = selector.transform(X_val)
    score = model.score(X_val_selected, y_val)
    print(f"Score with {num_features_to_select} features:", score)

    if score > best_score:
        best_score = score
        best_selected_features = selected_features

np.savetxt('best_selected_features_knn_selectkbest.txt', best_selected_features, fmt='%s')

print("Best score obtained:", best_score)
print("Number of features selected:", len(best_selected_features))
