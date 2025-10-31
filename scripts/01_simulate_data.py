import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import simulate_clinical_data, get_missing_summary, get_descriptive_stats

def main():
    """Main execution function."""
    # Simulate data
    print("\n[1/4] Simulating clinical dataset...")
    df = simulate_clinical_data(
        n_patients=600,
        missing_rate=0.25,
        random_seed=42
    )
    print(f"   ✓ Generated {len(df)} patient records")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    # Save raw data (use forward slashes or raw string)
    output_path = 'data/simulated_clinical_data.csv'  
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")
    
    # Generate missing data summary
    print("\n[2/4] Analyzing missing data patterns...")
    missing_summary = get_missing_summary(df)
    print(missing_summary.to_string(index=False))
    
    # Save missing summary
    missing_summary.to_csv('results/tables/missing_data_summary.csv', index=False)
    print("   ✓ Saved missing data summary")
    
    # Generate descriptive statistics
    print("\n[3/4] Computing descriptive statistics...")
    desc_stats = get_descriptive_stats(df)
    print(desc_stats)
    
    # Save descriptive stats
    desc_stats.to_csv('results/tables/descriptive_statistics.csv')
    print("   ✓ Saved descriptive statistics")
    
    # Display data preview
    print("\n[4/4] Data preview:")
    print(df.head(10).to_string(index=False))
    
    # Summary statistics for key variables
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patients: {len(df)}")
    print(f"\nDose distribution:")
    print(df['DOSE'].value_counts().sort_index())
    print(f"\nSex distribution:")
    print(df['SEX'].value_counts())
    print(f"\nResponse rate: {df['RESPONSE'].mean():.2%} (excluding missing)")
    print(f"\nAge range: {df['AGE'].min():.1f} - {df['AGE'].max():.1f} years")
    print(f"Weight range: {df['WT'].min():.1f} - {df['WT'].max():.1f} kg")
    
    print("\n✓ Data simulation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()