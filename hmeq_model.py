import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

# --- PARAMETERS ---
DATA_PATH = os.path.join('data', 'hmeq.csv')  # Adjust if your file is not in data/
MODEL_DIR = 'models'
K_BEST_FEATURES = 10
RANDOM_STATE = 42

# --- 1. LOAD DATASET ---
df = pd.read_csv(DATA_PATH)

# --- 2. BASIC CLEANING ---
cat_features = ["REASON", "JOB"]
num_features = [col for col in df.select_dtypes(include=[np.number]).columns if col != "BAD"]

# --- 3. IMPUTE MISSING VALUES ---
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='mean')

df[cat_features] = imputer_cat.fit_transform(df[cat_features])
df[num_features] = imputer_num.fit_transform(df[num_features])

# --- 4. ENCODE CATEGORICAL FEATURES ---
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cat = encoder.fit_transform(df[cat_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features), index=df.index)
df_full = pd.concat([df.drop(columns=cat_features), encoded_cat_df], axis=1)

# --- 5. FEATURE SELECTION ---
X = df_full.drop(columns=['BAD'])
y = df_full['BAD']

fs = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
X_selected = fs.fit_transform(X, y)
selected_features = X.columns[fs.get_support()]

# --- 6. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.33, random_state=RANDOM_STATE
)

# --- 7. TRAIN MODEL ---
clf = DecisionTreeClassifier(max_depth=100, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

print("Model trained on selected features:")
print(selected_features)

# --- 8. EVALUATE ---
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- 9. SAVE ALL OBJECTS ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(imputer_cat, os.path.join(MODEL_DIR, 'imputer_cat.joblib'))
joblib.dump(imputer_num, os.path.join(MODEL_DIR, 'imputer_num.joblib'))
joblib.dump(encoder, os.path.join(MODEL_DIR, 'encoder.joblib'))
joblib.dump(fs, os.path.join(MODEL_DIR, 'selector.joblib'))
joblib.dump(clf, os.path.join(MODEL_DIR, 'decision_tree.joblib'))
joblib.dump(selected_features, os.path.join(MODEL_DIR, 'selected_features.joblib'))

print(f"Saved all pipeline objects to ./{MODEL_DIR}/")
