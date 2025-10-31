import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, roc_curve, auc
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_data
from src.plotting import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance
)

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate model and return metrics."""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def main():
    """Main execution function."""
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    # Load imputed data
    print("\n[1/8] Loading imputed data...")
    df = load_data('data/imputed_data.csv')
    print(f"   ✓ Loaded {len(df)} records")
    
    # Prepare features and target
    print("\n[2/8] Preparing features and target...")
    
    # Encode SEX
    df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
    
    feature_cols = ['DOSE', 'AGE', 'WT', 'SEX_ENCODED', 'CMAX', 'AUC']
    X = df[feature_cols].values
    y = df['RESPONSE'].values
    
    print(f"   Features: {feature_cols}")
    print(f"   Shape: X={X.shape}, y={y.shape}")
    
    # Split data
    print("\n[3/8] Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Response rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Train Logistic Regression
    print("\n[4/8] Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        penalty='l2',
        C=1.0
    )
    lr_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores_lr = cross_val_score(
        lr_model, X_train, y_train, cv=5, scoring='roc_auc'
    )
    print(f"   Cross-validation AUC: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std():.3f})")
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics_lr = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression')
    print(f"   Test AUC: {metrics_lr['ROC_AUC']:.3f}")
    
    # Save model
    joblib.dump(lr_model, 'trained_models/logistic_regression_model.joblib')
    print("   ✓ Saved model")
    
    # Train Random Forest
    print("\n[5/8] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores_rf = cross_val_score(
        rf_model, X_train, y_train, cv=5, scoring='roc_auc'
    )
    print(f"   Cross-validation AUC: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std():.3f})")
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
    print(f"   Test AUC: {metrics_rf['ROC_AUC']:.3f}")
    
    # Save model
    joblib.dump(rf_model, 'trained_models/random_forest_model.joblib')
    print("   ✓ Saved model")
    
    # Create performance comparison table
    print("\n[6/8] Creating performance comparison...")
    performance_df = pd.DataFrame([metrics_lr, metrics_rf])
    print("\nModel Performance Comparison:")
    print(performance_df.to_string(index=False))
    
    # Save performance table
    performance_df.to_csv('results/tables/model_performance.csv', index=False)
    print("   ✓ Saved performance table")
    
    # Generate visualizations
    print("\n[7/8] Generating visualizations...")
    
    # ROC curves
    fig_roc_lr = plot_roc_curve(
        y_test, y_pred_proba_lr,
        model_name='Logistic Regression',
        save_path='results/figures/05_roc_curve_logistic.png'
    )
    plt.close(fig_roc_lr)
    
    fig_roc_rf = plot_roc_curve(
        y_test, y_pred_proba_rf,
        model_name='Random Forest',
        save_path='results/figures/06_roc_curve_randomforest.png'
    )
    plt.close(fig_roc_rf)
    print("   ✓ Saved ROC curves")
    
    # Confusion matrices
    fig_cm_lr = plot_confusion_matrix(
        y_test, y_pred_lr,
        model_name='Logistic Regression',
        save_path='results/figures/07_confusion_matrix_logistic.png'
    )
    plt.close(fig_cm_lr)
    
    fig_cm_rf = plot_confusion_matrix(
        y_test, y_pred_rf,
        model_name='Random Forest',
        save_path='results/figures/08_confusion_matrix_randomforest.png'
    )
    plt.close(fig_cm_rf)
    print("   ✓ Saved confusion matrices")
    
    # Feature importance (Logistic Regression coefficients)
    lr_coef = np.abs(lr_model.coef_[0])
    fig_imp_lr = plot_feature_importance(
        feature_cols,
        lr_coef,
        model_name='Logistic Regression (Coefficients)',
        top_n=len(feature_cols),
        save_path='results/figures/09_feature_importance_logistic.png'
    )
    plt.close(fig_imp_lr)
    
    # Feature importance (Random Forest)
    fig_imp_rf = plot_feature_importance(
        feature_cols,
        rf_model.feature_importances_,
        model_name='Random Forest',
        top_n=len(feature_cols),
        save_path='results/figures/10_feature_importance_randomforest.png'
    )
    plt.close(fig_imp_rf)
    print("   ✓ Saved feature importance plots")
    
    # Detailed classification reports
    print("\n[8/8] Generating detailed reports...")
    
    print("\nLogistic Regression - Classification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=['No Response', 'Response']))
    
    print("\nRandom Forest - Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['No Response', 'Response']))
    
    # Feature importance comparison
    print("\nFeature Importance Comparison:")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'LR_Coefficient': lr_coef,
        'RF_Importance': rf_model.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    print(importance_df.to_string(index=False))
    
    importance_df.to_csv('results/tables/feature_importance.csv', index=False)
    
    print("SUMMARY")
    print(f"Best Model: {'Random Forest' if metrics_rf['ROC_AUC'] > metrics_lr['ROC_AUC'] else 'Logistic Regression'}")
    print(f"Best AUC: {max(metrics_rf['ROC_AUC'], metrics_lr['ROC_AUC']):.3f}")
    print(f"\nModels saved in: models/")
    print(f"Results saved in: results/")
    
    print("\n✓ Model training completed successfully!")
   

if __name__ == "__main__":
    main()