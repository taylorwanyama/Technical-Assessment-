import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_data, split_features_target
from src.data_preprocessing import (
    ClinicalDataPreprocessor,
    create_before_after_comparison,
    remove_incomplete_cases
)
from src.plotting import plot_before_after_imputation

def main():
    """Main execution function."""
    print("=" * 60)
    print("DATA PREPROCESSING & IMPUTATION")
    print("=" * 60)
    
    # Create directories
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading simulated data...")
    df_raw = load_data('data/simulated_data.csv')
    print(f"   ✓ Loaded {len(df_raw)} records")
    
    # Remove cases with missing response (can't train without target)
    print("\n[2/6] Removing cases with missing RESPONSE...")
    df_clean = remove_incomplete_cases(df_raw, target_col='RESPONSE')
    print(f"   ✓ Retained {len(df_clean)} complete cases")
    
    # Separate ID column for later
    ids = df_clean['ID'].copy()
    df_for_imputation = df_clean.drop('ID', axis=1)
    
    # Store before imputation data
    df_before = df_for_imputation.copy()
    
    # Initialize preprocessor with MICE
    print("\n[3/6] Initializing MICE imputation...")
    preprocessor = ClinicalDataPreprocessor(
        imputation_method='mice',
        random_state=42
    )
    
    # Separate features and target
    X = df_for_imputation.drop('RESPONSE', axis=1)
    y = df_for_imputation['RESPONSE']
    
    # Fit and transform features
    print("   Processing features (this may take a moment)...")
    X_imputed = preprocessor.fit_transform(X, scale=False)
    
    # Combine back with target and ID
    df_imputed = X_imputed.copy()
    df_imputed['RESPONSE'] = y.values
    df_imputed['ID'] = ids.values
    
    # Reorder columns to match original
    cols_order = ['ID'] + [col for col in df_clean.columns if col != 'ID']
    df_imputed = df_imputed[cols_order]
    
    print("   ✓ Imputation completed")
    
    # Save imputed data
    print("\n[4/6] Saving imputed data...")
    output_path = 'data/imputed_data.csv'
    df_imputed.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")
    
    # Create before/after comparison
    print("\n[5/6] Creating before/after comparison...")
    comparison_vars = ['DOSE', 'AGE', 'WT', 'CMAX', 'AUC']
    
    # Comparison table
    comparison_table = create_before_after_comparison(
        df_before,
        df_imputed.drop(['ID', 'RESPONSE'], axis=1),
        comparison_vars
    )
    
    print("\nBefore vs After Imputation Statistics:")
    print(comparison_table.to_string(index=False))
    
    # Save comparison table
    comparison_table.to_csv(
        'results/tables/imputation_comparison.csv',
        index=False
    )
    print("   ✓ Saved comparison table")
    
    # Plot before/after distributions
    print("\n[6/6] Creating visualization...")
    imputation_vars = ['CMAX', 'AUC']
    fig_comparison = plot_before_after_imputation(
        df_before,
        df_imputed.drop(['ID', 'RESPONSE'], axis=1),
        columns=imputation_vars,
        save_path='results/figures/04_before_after_imputation.png'
    )
    plt.close(fig_comparison)
    print("   ✓ Saved: results/figures/04_before_after_imputation.png")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("IMPUTATION SUMMARY")
    print("=" * 60)
    print(f"Records before: {len(df_raw)}")
    print(f"Records after removing missing RESPONSE: {len(df_clean)}")
    print(f"Final imputed records: {len(df_imputed)}")
    print(f"\nMissing values remaining: {df_imputed.isnull().sum().sum()}")
    
    # Verify no missing values in key columns
    key_cols = ['CMAX', 'AUC', 'RESPONSE']
    print(f"\nMissing values in key variables:")
    for col in key_cols:
        n_missing = df_imputed[col].isnull().sum()
        print(f"  {col}: {n_missing}")
    
    print("\n✓ Preprocessing completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()