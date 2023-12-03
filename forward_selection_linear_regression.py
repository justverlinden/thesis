import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\code\my_data_encoded.csv'
df_encoded = pd.read_csv(file_path)

X = df_encoded.drop(columns=['funding_total_usd'])
y = df_encoded['funding_total_usd']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

best_score = float('-inf')
best_selected_features = []
remaining_features = list(X.columns)

threshold = 0.0001 

iteration = 1
while remaining_features:
    improved = False
    selected_feature = None
    
    for feature in remaining_features:
        features_to_try = best_selected_features + [feature]
        
        model.fit(X_train[features_to_try], y_train)
        score = model.score(X_val[features_to_try], y_val)
        
        if score - best_score > threshold:
            best_score = score
            selected_feature = feature
            improved = True
    
    if improved:
        best_selected_features.append(selected_feature)
        remaining_features.remove(selected_feature)
        print(f"Iteration {iteration}: Added '{selected_feature}', Score: {best_score}")
    else:
        break 
    
    iteration += 1

np.savetxt('best_selected_features_forward_selection.txt', best_selected_features, fmt='%s')

print("Best score obtained:", best_score)
print("Number of features selected:", len(best_selected_features))
