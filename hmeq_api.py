from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI()

# --- Load model pipeline objects ---
MODEL_DIR = 'models'
imputer_cat = joblib.load(os.path.join(MODEL_DIR, 'imputer_cat.joblib'))
imputer_num = joblib.load(os.path.join(MODEL_DIR, 'imputer_num.joblib'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.joblib'))
selector = joblib.load(os.path.join(MODEL_DIR, 'selector.joblib'))
clf = joblib.load(os.path.join(MODEL_DIR, 'decision_tree.joblib'))
selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.joblib'))

cat_features = ["REASON", "JOB"]
num_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

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

@app.post("/predict/")
def predict_loan(applicant: dict):
    df = pd.DataFrame([applicant])
    X_pred = preprocess_input(df)
    prediction = clf.predict(X_pred)
    return {"prediction": int(prediction[0])}
