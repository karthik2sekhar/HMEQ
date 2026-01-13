# HMEQ Loan Default Prediction

A machine learning project for predicting home equity loan defaults using the HMEQ dataset. This project includes data preprocessing, model training, and a FastAPI-based REST API for real-time predictions.

## 📊 Project Overview

The **Home Equity (HMEQ)** dataset contains baseline and loan performance information for home equity loans. The target variable `BAD` indicates whether an applicant eventually defaulted (1) or paid back (0) their loan.

### Key Features Used

| Feature | Description | Importance |
|---------|-------------|------------|
| `DELINQ` | Number of delinquent credit lines | Strong correlation with default |
| `DEROG` | Number of major derogatory reports | Strong correlation |
| `DEBTINC` | Debt-to-income ratio | Key indicator for defaulters |
| `CLAGE` | Age of oldest credit line (months) | Negative correlation - credit history matters |
| `JOB` | Job category | Default rate varies by occupation |
| `REASON` | Loan purpose (DebtCon/HomeImp) | Engineered based on testing |
| `LOAN` | Loan amount requested | Weak direct relationship |
| `VALUE` | Current property value | Weak direct relationship |
| `MORTDUE` | Amount due on existing mortgage | Weak direct relationship |
| `YOJ` | Years at present job | Negligible relationship |
| `NINQ` | Number of recent credit inquiries | Weak correlation |
| `CLNO` | Number of credit lines | Weak correlation |

## 🏗️ Project Structure

```
HMEQ/
├── data/                    # Dataset directory
│   └── hmeq.csv            # HMEQ dataset
├── models/                  # Trained model artifacts
│   ├── imputer_cat.joblib  # Categorical imputer
│   ├── imputer_num.joblib  # Numerical imputer
│   ├── encoder.joblib      # OneHot encoder
│   ├── selector.joblib     # Feature selector
│   ├── decision_tree.joblib # Trained classifier
│   └── selected_features.joblib
├── notebooks/               # Jupyter notebooks for EDA
│   └── hmeq-model.ipynb
├── hmeq_model.py           # Model training script
├── hmeq_api.py             # FastAPI prediction service
├── predict_hmeq.py         # Batch prediction script
├── Feature-Include-Why.csv # Feature importance documentation
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthik2sekhar/HMEQ.git
   cd HMEQ
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   
   Place the `hmeq.csv` file in the `data/` directory.

### Training the Model

Run the training script to create model artifacts:

```bash
python hmeq_model.py
```

This will:
- Load and preprocess the HMEQ dataset
- Impute missing values (mean for numerical, mode for categorical)
- Encode categorical features using OneHotEncoder
- Select top 10 features using SelectKBest (ANOVA F-score)
- Train a Decision Tree Classifier
- Save all pipeline objects to the `models/` directory

### Making Predictions

#### Option 1: Batch Prediction

```bash
python predict_hmeq.py
```

#### Option 2: REST API

Start the FastAPI server:

```bash
uvicorn hmeq_api:app --reload
```

The API will be available at `http://127.0.0.1:8000`

**Example API Request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "LOAN": 25000,
    "MORTDUE": 50000,
    "VALUE": 75000,
    "REASON": "DebtCon",
    "JOB": "Office",
    "YOJ": 5,
    "DEROG": 0,
    "DELINQ": 0,
    "CLAGE": 120,
    "NINQ": 1,
    "CLNO": 10,
    "DEBTINC": 35.0
  }'
```

**Response:**
```json
{"prediction": 0}
```
- `0` = Loan likely to be repaid
- `1` = Loan likely to default

## 🔧 Model Pipeline

1. **Imputation**: Missing values handled using SimpleImputer
   - Categorical: Most frequent value
   - Numerical: Mean value

2. **Encoding**: OneHotEncoder for categorical features (`REASON`, `JOB`)

3. **Feature Selection**: SelectKBest with ANOVA F-score (k=10)

4. **Classification**: Decision Tree Classifier (max_depth=100)

## 📈 Model Performance

The model is evaluated using standard classification metrics including precision, recall, and F1-score on a 33% held-out test set.

## 🛠️ Technologies Used

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML algorithms & preprocessing
- **joblib** - Model serialization
- **FastAPI** - REST API framework
- **uvicorn** - ASGI server

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
