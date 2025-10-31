import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_data
from src.plotting import (
    plot_missing_data,
    plot_distributions,
    plot_correlation_matrix
)

def main():
    """Main execution function."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Create results directories
    os.makedirs('results/figures', exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading simulated data...")
    df = load_data('data/simulated_data.csv')
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} variables")
    
    # Plot missing data patterns
    print("\n[2/4] Visualizing missing data patterns...")
    fig_missing = plot_missing_data(
        df,
        save_path='results/figures/01_missing_data_pattern.png'
    )
    plt.close(fig_missing)
    print("   ✓ Saved: results/figures/01_missing_data_pattern.png")
    
    # Plot distributions of key variables
    print("\n[3/4] Plotting variable distributions...")
    numeric_vars = ['DOSE', 'AGE', 'WT', 'CMAX', 'AUC', 'RESPONSE']
    fig_dist = plot_distributions(
        df,
        columns=numeric_vars,
        save_path='results/figures/02_variable_distributions.png'
    )
    plt.close(fig_dist)
    print("   ✓ Saved: results/figures/02_variable_distributions.png")
    
    # Plot correlation matrix
    print("\n[4/4] Creating correlation matrix...")
    fig_corr = plot_correlation_matrix(
        df,
        save_path='results/figures/03_correlation_matrix.png'
    )
    plt.close(fig_corr)
    print("   ✓ Saved: results/figures/03_correlation_matrix.png")
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Correlation with response
    print("\nCorrelations with RESPONSE:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numeric_cols].corr()['RESPONSE'].sort_values(ascending=False)
    print(correlations[correlations.index != 'RESPONSE'])
    
    # Group statistics by dose
    print("\n\nMean PK parameters by dose level:")
    dose_stats = df.groupby('DOSE')[['CMAX', 'AUC']].mean()
    print(dose_stats)
    
    # Response rate by sex
    print("\n\nResponse rate by sex:")
    response_by_sex = df.groupby('SEX')['RESPONSE'].agg(['mean', 'count'])
    response_by_sex.columns = ['Response_Rate', 'N']
    print(response_by_sex)
    
    print("\n✓ Exploratory data analysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()