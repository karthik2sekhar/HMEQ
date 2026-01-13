import pandas as pd
import joblib
import os
import numpy as np

# --- Load pipeline objects ---
MODEL_DIR = 'models'
imputer_cat = joblib.load(os.path.join(MODEL_DIR, 'imputer_cat.joblib'))
imputer_num = joblib.load(os.path.join(MODEL_DIR, 'imputer_num.joblib'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.joblib'))
selector = joblib.load(os.path.join(MODEL_DIR, 'selector.joblib'))
clf = joblib.load(os.path.join(MODEL_DIR, 'decision_tree.joblib'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.joblib'))

cat_features = ["REASON", "JOB"]
num_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']  # update if needed

def preprocess_input(df):
    df = df.copy()
    df[cat_features] = imputer_cat.transform(df[cat_features])
    df[num_features] = imputer_num.transform(df[num_features])
    encoded_cat = encoder.transform(df[cat_features])
    encoded_cat_df = pd.DataFrame(
        encoded_cat, columns=encoder.get_feature_names_out(cat_features), index=df.index
    )
    df_full = pd.concat([df.drop(columns=cat_features), encoded_cat_df], axis=1)
    df_selected = df_full[selected_features]
    return df_selected

# --- Example usage ---
# Load new data to predict, or manually construct a DataFrame:
new_data = pd.read_csv('data/hmeq.csv').iloc[[4]]    # You can swap in new data here

X_pred = preprocess_input(new_data)
prediction = clf.predict(X_pred)

print(f"Prediction (BAD, 1=default): {prediction[0]}")
