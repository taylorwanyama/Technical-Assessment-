import pandas as pd
import numpy as np
from typing import Tuple, Optional

def simulate_clinical_data(
    n_patients: int = 600,
    missing_rate: float = 0.25,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Simulate a clinical dataset.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to simulate
    missing_rate : float
        Proportion of missing data in key variables
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame(Simulated clinical dataset)
        Simulated clinical dataset
    """
    np.random.seed(random_seed)
    
    # Generate patient IDs
    ids = [f"PT{str(i).zfill(4)}" for i in range(1, n_patients + 1)]
    
    # Simulate demographics
    age = np.random.normal(55, 15, n_patients).clip(18, 85)
    weight = np.random.normal(75, 15, n_patients).clip(45, 120)
    sex = np.random.choice(['M', 'F'], n_patients, p=[0.4, 0.6])
    
    # Simulate dosing (3 dose levels)
    dose = np.random.choice([100, 200, 400], n_patients, p=[0.3, 0.4, 0.3])
    
    # Simulate PK parameters with realistic relationships
    cmax_base = (dose / weight) * 10
    cmax_noise = np.random.normal(0, cmax_base * 0.3, n_patients)
    age_effect = (age - 55) * 0.05
    cmax = (cmax_base + cmax_noise - age_effect).clip(1, None)
    auc_base = cmax * (8 + np.random.normal(0, 2, n_patients))
    weight_effect = (75 - weight) * 0.5
    auc = (auc_base + weight_effect).clip(10, None)
    
    # Logistic relationship
    linear_predictor = (
        -5 +
        0.02 * auc +
        0.01 * dose -
        0.02 * age +
        0.5 * (sex == 'M')
    )
    response_prob = 1 / (1 + np.exp(-linear_predictor))
    response = (np.random.random(n_patients) < response_prob).astype(int)
    
    # Create dataframe
    df = pd.DataFrame({
        'ID': ids,
        'DOSE': dose,
        'AGE': age,
        'WT': weight,
        'SEX': sex,
        'CMAX': cmax,
        'AUC': auc,
        'RESPONSE': response
    })
    
    # Introduce missingness (Missing Completely At Random)
    n_missing = int(n_patients * missing_rate)
    
    # Create missingness in CMAX
    missing_cmax_idx = np.random.choice(n_patients, n_missing, replace=False)
    df.loc[missing_cmax_idx, 'CMAX'] = np.nan
    
    # Create missingness in AUC
    missing_auc_idx = np.random.choice(n_patients, n_missing, replace=False)
    df.loc[missing_auc_idx, 'AUC'] = np.nan
    
    # Create missingness in RESPONSE
    n_missing_response = int(n_patients * missing_rate * 0.8)
    missing_response_idx = np.random.choice(
        n_patients, n_missing_response, replace=False
    )
    df.loc[missing_response_idx, 'RESPONSE'] = np.nan
    
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load clinical data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    df = pd.read_csv(filepath)  
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary of missing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values per column
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    summary = pd.DataFrame({
        'Variables': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_pct.values
    })
    
    summary = summary[summary['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )
    
    return summary


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Descriptive statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    
    # Additional statistics
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    
    return stats.round(2)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = 'RESPONSE',
    id_col: str = 'ID'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into features, target, and IDs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    id_col : str
        Name of ID column
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        Features, target, and IDs
    """
    # Extract IDs
    ids = df[id_col]
    
    # Extract target
    target = df[target_col]
    
    # Extract features (exclude ID and target)
    feature_cols = [col for col in df.columns if col not in [id_col, target_col]]
    features = df[feature_cols]
    
    return features, target, ids
