import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

class ClinicalDataPreprocessor:
    """
    Preprocessor for clinical trial data with imputation and encoding.
    """
    
    def __init__(self, imputation_method: str = 'mice', random_state: int = 42):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        imputation_method : str
            Method for imputation ('mice', 'mean', 'median')
        random_state : int
            Random seed for reproducibility
        """
        self.imputation_method = imputation_method
        self.random_state = random_state
        self.numeric_imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        scale: bool = True
    ) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        scale : bool
            Whether to scale numeric features
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        self.feature_names = numeric_cols + categorical_cols
        
        # Encode categorical variables
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            non_null_mask = df_processed[col].notna()
            if non_null_mask.sum() > 0:
                df_processed.loc[non_null_mask, col] = self.label_encoders[col].fit_transform(
                    df_processed.loc[non_null_mask, col]
                )
                df_processed[col] = df_processed[col].astype(float)
        
        # Initialize imputer based on method for numerical variables
        if self.imputation_method == 'mice':
            self.numeric_imputer = IterativeImputer(
                random_state=self.random_state,
                max_iter=10,
                verbose=0
            )
        elif self.imputation_method == 'mean':
            self.numeric_imputer = SimpleImputer(strategy='mean')
        elif self.imputation_method == 'median':
            self.numeric_imputer = SimpleImputer(strategy='median')
        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")
        
        all_cols = numeric_cols + categorical_cols
        df_processed[all_cols] = self.numeric_imputer.fit_transform(df_processed[all_cols])
        
        # Scale numeric features if requested
        if scale and len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        return df_processed
    
    def transform(
        self,
        df: pd.DataFrame,
        scale: bool = True
    ) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        scale : bool
            Whether to scale numeric features
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe
        """
        if self.numeric_imputer is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in self.label_encoders:
                non_null_mask = df_processed[col].notna()
                if non_null_mask.sum() > 0:
                    df_processed.loc[non_null_mask, col] = self.label_encoders[col].transform(
                        df_processed.loc[non_null_mask, col]
                    )
                    df_processed[col] = df_processed[col].astype(float)
        
        # Impute
        all_cols = numeric_cols + categorical_cols
        df_processed[all_cols] = self.numeric_imputer.transform(df_processed[all_cols])
        
        # Scale if requested
        if scale and self.scaler is not None and len(numeric_cols) > 0:
            df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
        
        return df_processed


def create_before_after_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Create comparison of statistics before and after imputation.
    
    Parameters:
    -----------
    df_before : pd.DataFrame
        Data before imputation
    df_after : pd.DataFrame
        Data after imputation
    columns : list
        Columns to compare
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for col in columns:
        if col in df_before.columns and col in df_after.columns:
            before_stats = {
                'Variable': col,
                'Stage': 'Before',
                'Mean': df_before[col].mean(),
                'Std': df_before[col].std(),
                'Missing': df_before[col].isnull().sum(),
                'Missing_Pct': (df_before[col].isnull().sum() / len(df_before)) * 100
            }
            
            after_stats = {
                'Variable': col,
                'Stage': 'After',
                'Mean': df_after[col].mean(),
                'Std': df_after[col].std(),
                'Missing': df_after[col].isnull().sum(),
                'Missing_Pct': (df_after[col].isnull().sum() / len(df_after)) * 100
            }
            
            comparison.extend([before_stats, after_stats])
    
    return pd.DataFrame(comparison).round(3)


def remove_incomplete_cases(
    df: pd.DataFrame,
    target_col: str = 'RESPONSE'
) -> pd.DataFrame:
    """
    Remove cases with missing target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with complete cases for target
    """
    df_complete = df[df[target_col].notna()].copy()
    print(f"Removed {len(df) - len(df_complete)} cases with missing {target_col}")
    return df_complete