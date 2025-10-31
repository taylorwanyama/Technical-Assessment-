import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, confusion_matrix
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def plot_missing_data(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize missing data patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Missing data heatmap
    missing_data = df.isnull()
    sns.heatmap(
        missing_data.T,
        cbar=True,
        yticklabels=df.columns,
        cmap='YlOrRd',
        ax=ax1
    )
    ax1.set_title('Missing Data Pattern', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Patient Index')
    
    # Missing data bar plot
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    
    missing_pct.plot(kind='barh', color='coral', ax=ax2)
    ax2.set_title('Missing Data Percentage by Variable', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Percentage Missing (%)')
    ax2.set_ylabel('Variable')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of numeric variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to plot
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            data = df[col].dropna()
            
            axes[idx].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(data.mean(), color='red', linestyle='--', 
                             label=f'Mean: {data.mean():.2f}')
            axes[idx].legend()
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix for numeric variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_before_after_imputation(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare distributions before and after imputation.
    
    Parameters:
    -----------
    df_before : pd.DataFrame
        Data before imputation
    df_after : pd.DataFrame
        Data after imputation
    columns : List[str]
        Columns to compare
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
    
    if n_cols == 1:
        axes = [axes]
    
    for idx, col in enumerate(columns):
        if col in df_before.columns and col in df_after.columns:
            # Plot before
            axes[idx].hist(
                df_before[col].dropna(),
                bins=30,
                alpha=0.5,
                label='Before Imputation',
                color='coral',
                edgecolor='black'
            )
            
            # Plot after
            axes[idx].hist(
                df_after[col],
                bins=30,
                alpha=0.5,
                label='After Imputation',
                color='skyblue',
                edgecolor='black'
            )
            
            axes[idx].set_title(
                f'{col} - Before vs After Imputation',
                fontweight='bold'
            )
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of model
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of model
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_xticklabels(['No Response', 'Response'])
    ax.set_yticklabels(['No Response', 'Response'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = 'Model',
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : List[str]
        Names of features
    importances : np.ndarray
        Feature importance values
    model_name : str
        Name of model
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(
        range(len(indices)),
        importances[indices],
        color='steelblue',
        edgecolor='black'
    )
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(
        f'Top {top_n} Feature Importances - {model_name}',
        fontsize=14,
        fontweight='bold'
    )
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig