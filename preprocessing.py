import pandas as pd
import numpy as np
from collections import Counter

def hierarchical_random_sampling_imputation(df, higher_level_cols, lower_level_col):
    for higher_level_col in higher_level_cols:
        missing_indices = df[df[lower_level_col].isnull()].index
        for idx in missing_indices:
            same_higher_level = df[df[higher_level_col] == df.loc[idx, higher_level_col]][lower_level_col]
            if not same_higher_level.isnull().all():
                non_missing_values = same_higher_level.dropna().values
                random_value = np.random.choice(non_missing_values)
                df.loc[idx, lower_level_col] = random_value
    return df

def perform_encoding(df):
    categorical_columns = ['category_list', 'country_code', 'state_code', 'region', 'city']
    encoded_dfs = []

    for col in categorical_columns:
        encoded_cols = pd.get_dummies(df[col], prefix=col)
        top_categories = encoded_cols.columns[df[col].value_counts(normalize=True).cumsum() <= 0.8]

        encoded_df = encoded_cols[top_categories]
        encoded_dfs.append(encoded_df)

    status_df = pd.get_dummies(df['status'], prefix='status')
    encoded_dfs.append(status_df)

    df = pd.concat([df] + encoded_dfs, axis=1)
    df.drop(columns=categorical_columns + ['status'], inplace=True)

    return df


if __name__ == '__main__':
    file_path = r'C:\Users\just-\Documents\Tilburg University\Master thesis\big_startup_secsees_dataset.csv'
    df = pd.read_csv(file_path)

    df = df.drop(columns=['permalink', 'name', 'homepage_url'])

    # Remove rows with missing data for the target variable
    df['funding_total_usd'].replace("-", np.nan, inplace=True)
    df.dropna(subset=['funding_total_usd'], inplace=True)
    df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
    
    # Convert date columns to dates
    date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']
    df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')

    # Missing founded dates are replaced with the day they received their first funding
    df['founded_at'].fillna(df['first_funding_at'], inplace=True)
    
    invalid_dates = df[df[date_columns].isnull().any(axis=1)]
    future_dates = df[(df[date_columns[0]] > pd.Timestamp.now()) |
                  (df[date_columns[1]] > pd.Timestamp.now()) |
                  (df[date_columns[2]] > pd.Timestamp.now())]
    
    # Drop rows with missing dates for the first funding, and drop rows with dates in the future
    df.dropna(subset=['first_funding_at'], inplace=True)
    df.drop(index=future_dates.index, inplace=True)

    # Create new feature with days in between first and last funding
    df['days_between_fundings'] = (df['last_funding_at'] - df['first_funding_at']).dt.days

    # Convert date columns to year values
    df['founded_at'] = pd.to_datetime(df['founded_at']).dt.year
    df['first_funding_at'] = pd.to_datetime(df['first_funding_at']).dt.year
    df['last_funding_at'] = pd.to_datetime(df['last_funding_at']).dt.year

    # Apply hierarchical random sampling to location columns
    hierarchy = [('country_code', 'state_code'), ('state_code', 'region'), ('region', 'city')]

    for higher, lower in hierarchy:
        df = hierarchical_random_sampling_imputation(df, [higher], lower)

    # Remove outliers based on a threshold value of 3 for 'funding_total_usd'
    Q1 = df['funding_total_usd'].quantile(0.25)
    Q3 = df['funding_total_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df = df[(df['funding_total_usd'] >= lower_bound) & (df['funding_total_usd'] <= upper_bound)]

    # Pick the first category for rows with multiple categories, and missing values get 'Other' as category
    df['category_list'] = df['category_list'].str.split('|').str[0]
    df['category_list'] = df['category_list'].fillna('Other')

    categorical_columns = ['category_list', 'status', 'country_code', 'state_code', 'region', 'city']
    for col in categorical_columns:
        value_counts = df[col].value_counts(normalize=True)
        cumulative_percentage = value_counts.cumsum()
        categories_required = cumulative_percentage[cumulative_percentage <= 0.8].count()
        print(f"Number of categories in '{col}' to represent 90% of the data: {categories_required}")

    # Perform encoding
    df_encoded = perform_encoding(df)
    pd.set_option('display.max_rows', 100)
    print(df_encoded.head(100))

    # After your preprocessing steps and before modeling, assuming df_encoded contains the predictors
    X = df_encoded.drop(columns=['funding_total_usd'])  

    # Assuming X contains the predictor variables after preprocessing
    correlation_matrix = X.corr()

    # Create a correlation matrix and check for high correlations between predictors
    high_correlations = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:  
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                high_correlations.add((colname_i, colname_j))

    if high_correlations:
        print("Highly correlated features:")
        print(high_correlations)
    else:
        print("No highly correlated features found.")

    columns_to_drop = [col for col, _ in high_correlations]
    df_encoded.drop(columns=columns_to_drop, inplace=True)


    df_encoded.to_csv('my_data_encoded.csv', index=False) 



