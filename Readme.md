# Clinical PK/PD Modeling Assessment

## Project Overview

This project analyzes a simulated clinical dataset to predict patient response to treatment. The workflow includes data simulation, comprehensive exploratory data analysis, missing data imputation, and predictive modeling.

**Objective:** To build and compare interpretable and machine learning models to predict binary clinical response based on patient characteristics and drug exposure.

---

## Dataset Description

### Simulated Clinical Trial Data
- **Sample Size:** 600 subjects
- **Study Design:** Simulated Phase 2/3 clinical trial with three dose levels

### Variables

| Variable | Type | Description | Range/Values |
|----------|------|-------------|--------------|
| ID | Integer | Subject identifier | 1-600 |
| DOSE | Numeric | Drug dose (mg) |  100, 200, 400 |
| AGE | Numeric | Patient age (years) | 18-85 |
| WT | Numeric | Body weight (kg) | 45-120 |
| SEX | Categorical | Biological sex | M, F |
| CMAX | Numeric | Maximum plasma concentration | Continuous |
| AUC | Numeric | Area under concentration-time curve | Continuous |
| RESPONSE | Binary | Clinical response | 0 (Non-Responder), 1 (Responder) |

### Missing Data Pattern
- **Missingness Type:** Missing Completely At Random
- **Affected Variables:** CMAX, AUC, RESPONSE
- **Missingness Rate:** ~25% per variable
- **Rationale:** Simulates realistic clinical scenarios 
---

## Project Workflow

### 1. Data Simulation (`01_simulate_data.py`)
- Simulates realistic PK/PD relationships:
  - CMAX depends on DOSE and body weight
  - AUC correlates with CMAX
  - RESPONSE follows logistic exposure-response relationship
- Introduces 25% MCAR missingness
- **Output:** `data/simulated_data.csv`

### 2. Exploratory Data Analysis (`02_run_eda.py`)
- **Missing data analysis:**
  - Quantitative summary tables
  - Missingness visualization (matrix, heatmap)
- **Distribution analysis:**
  - Histograms with KDE
  - Boxplots for outlier detection
- **Correlation analysis:**
  - Heatmaps showing variable relationships
- **Outputs:** 
  - 5 figures in `results/figures/`
  - 3 summary tables in `results/tables/`

### 3. Data Preprocessing (`03_preprocess.py`)
- **Missing data imputation:**
  - Method: MICE (Multiple Imputation by Chained Equations)
  - Iterations: 10 (converged)
  - Quality validation: Distribution comparison plots
- **Feature engineering:**
  - Categorical encoding: SEX (M=1, F=0)
  - Feature scaling: StandardScaler
- **Data splitting:**
  - 80% training (480 samples)
  - 20% test (120 samples)
  - Stratified by RESPONSE to maintain class balance
- **Outputs:**
  - `data/imputed_data.csv`
  - `models/imputer.pkl`, `models/scaler.pkl`
  - Imputation quality figures

### 4. Model Training (`04_train_models.py`)
Two models trained and compared:

#### Model 1: Logistic Regression
#### Model 2: Random Forest Classifier


### 5. Model Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Diagnostics:** ROC curves, confusion matrices, classification reports
- **Outputs:**
  - `results/tables/model_performance.csv`
  - ROC curves and comparison plots

### 6. API Deployment 
- **Framework:** FastAPI
- **Endpoints:**
  - `POST /predict`: Make predictions
  - `GET /health`: Health check
- **Features:**
  - Input validation
  - Automatic preprocessing
  - Probability outputs

---

## Installation & Setup

### Requirements
- Python 3.8-3.12
- See `requirements.txt` for package versions

### Setup Instructions
```bash
# 1. Clone repository
git clone https://github.com/taylorwanyama/Technical-Assessment-
cd project_root

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Complete Pipeline (Recommended)
```bash
# Step 1: Simulate data
python src\data_utils.py
python scripts/01_simulate_data.py

# Step 2: Exploratory analysis
python scripts/02_run_eda.py

# Step 3: Preprocessing
python scripts/03_preprocess.py

# Step 4: Model training
python scripts/04_train_models.py

# Step 5: API 
python scripts/05_api.py
```
---

## Results Summary

### Model Performance

[Note: Results will vary based on random seed and simulation]

**Model Comparison:**
| Model | Accuracy | Precision | Recall | F1_Score | ROC_AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.8854 |      0.9038 |   0.8868 |     0.8952 |    0.9544 |
| Random Forest Classifier |     0.8750 |      0.8868 |   0.8868 |     0.8868 |    0.9478 |

### Key Findings

1. **Imputation Quality:**
   - MICE successfully imputed all missing values
   - Post-imputation distributions closely match observed data
   - No evidence of imputation-induced bias

2. **Feature Importance:**
   - AUC is the single most important feature, followed by CMAX
   - DOSE ranks third in importance
   -AGE is the most important non-PK feature while WT and SEX have minimal importance

---

## Project Structure
```
project_root/
│
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
│
├── notebooks/                  # Interactive development and analysis
│   ├── EDA.ipynb               # Detailed Exploratory Data Analysis
│   └── modeling.ipynb          # Model selection and evaluation
│
├── src/                        # Reusable core code modules
│   ├── __init__.py
│   ├── data_utils.py           # Data loading, simulation, and statistics
│   ├── data_preprocessing.py   # Imputation, scaling, and encoding logic
│   └── plotting.py             # Visualization functions
│
├── scripts/                    # Automated execution pipeline
│   ├── 01_simulate_data.py     # Generate raw data
│   ├── 02_run_eda.py           # Run visualization and initial analysis
│   ├── 03_preprocess.py        # Execute data cleaning and transformation
│   ├── 04_train_models.py      # Train and save models
│   └── 05_api.py               # FastAPI prediction service
│
├── data/                       # Input and intermediate data files
│   ├── simulated_data.csv
│   └── imputed_data.csv
│
├── results/                    # Outputs from the pipeline
│   ├── figures/                # Saved plots (missing data, ROC, etc.)
│   └── tables/
│       └── model_performance.csv # Model metrics summary
│
├── models/                     # Trained and serialized models (.joblib)
│   ├── logistic_model.joblib
│   └── randomforest_model.joblib
│
└── .git ignore                 # Files to ignore in Git
```

---

## Key Methodological Decisions

### 1. Missing Data Handling
**Choice: MICE (Multiple Imputation by Chained Equations)**
- **Rationale:** Preserves distributions and correlations 
- **Alternative considered:** Mean and Median Imputation
- **Validation:** Distribution comparison plots confirm imputation quality

### 2. Model Selection
**Choice: Logistic Regression + Random Forest**
- **Logistic Regression:**
  - Meets interpretability requirement
  - Provides coefficient-based insights
  - Fast training and prediction
- **Random Forest:**
  - State-of-practice ML algorithm
  - Handles non-linear relationships
  - Provides feature importance
- **Alternatives considered:** XGBoost 

### 3. Train/Test Split
**Choice: 80/20 stratified split**
- **Rationale:** Standard for datasets of this size
- **Stratification:** Maintains class balance in both sets
- **No validation set:** Would leave too few test samples (150 total)

### 4. Feature Scaling
**Choice: StandardScaler (Z-score normalization)**
- **Rationale:** Required for logistic regression with regularization
- **Alternative considered:** MinMaxScaler (less robust to outliers)

---

## Limitations & Future Work

### Current Limitations
1. **Simulated data:** Real clinical data has more complex relationships
2. **MCAR assumption:** Real missingness often MAR or MNAR
3. **Binary outcome:** Real clinical endpoints may be continuous or time-to-event

### Potential Improvements
1. **Additional features:** Include biomarkers, genetic data, concomitant medications
2. **Advanced imputation:** Sensitivity analysis with different missingness assumptions
3. **Model ensemble:** Combine multiple models for robust predictions
4. **External validation:** Test on independent dataset
5. **Longitudinal modeling:** Incorporate repeated measurements over time

---

## API Usage 

### Start API Server
```bash
python scripts/05_api.py
```

### Make Predictions

**Using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/docs#/default/predict_predict_post" \
  -H "Content-Type: application/json" \
  -d '{
  "DOSE": 200,
  "AGE": 55,
  "WT": 75,
  "SEX": "F",
  "CMAX": 25.5,
  "AUC": 180.3
}'
```

**Expected Response:**
```json
{
  "patient_data": {
    "DOSE": 200,
    "AGE": 55,
    "WT": 75,
    "SEX": "F",
    "CMAX": 25.5,
    "AUC": 180.3
  },
  "logistic_regression": {
    "prediction": 0,
    "prediction_label": "No Response",
    "probability_no_response": 0.6626,
    "probability_response": 0.3374
  }
}
```
---

## Reproducibility

All results are reproducible using:
- Random seed: 42 (used throughout)
- Package versions: See `requirements.txt`

To reproduce exactly:
```bash
python src\data_utils.py
python scripts/01_simulate_data.py  
python scripts/02_run_eda.py
python scripts/03_preprocess.py     
python scripts/04_train_models.py   
```

---
