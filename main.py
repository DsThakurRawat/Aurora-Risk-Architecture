import sys
import os
import numpy as np
from src.data_pipeline.data_loader import SyntheticDataSimulator
from src.models.logistic_regression import LogisticRegressionModel
from src.fairness.fairness_metrics import FairnessMetrics

def main():
    print("--- Aurora Risk Architecture: Fair Credit Scoring Prototype ---\\n")
    
    print("1. Simulating Unbanked Borrower Demographics...")
    simulator = SyntheticDataSimulator(n_samples=5000, random_state=42)
    df = simulator.generate()
    
    # Exclude Protected Attribute A from training (Blind training)
    features = ["inflow_consistency", "savings_buffer_days", "phone_tenure_months", "utility_timeliness"]
    X = df[features].values
    y = df["Y"].values
    A = df["A"].values
    
    print(f"   Features shapes: {X.shape}. Simulated True Repay Rate: {np.mean(y == 1):.2%}")
    print(f"   Group 0 (Unprivileged) Representation: {np.mean(A == 0):.2%}")
    print(f"   Group 1 (Privileged) Representation: {np.mean(A == 1):.2%}\\n")
    
    print("2. Training Baseline Transparent Model (Logistic Regression without sklearn)...")
    # Standardize features for gradient descent
    def standardize(X):
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    X_scaled = standardize(X)

    model = LogisticRegressionModel(learning_rate=0.01, epochs=1000, lambda_reg=0.1)
    
    # Base training without sample weights
    model.fit(X_scaled, y)
    
    print("3. Assessing Fairness Capabilities on Global Decision Threshold (0.5)...")
    
    predictions = model.predict(X_scaled, threshold=0.5)
    
    fairness_report = FairnessMetrics.evaluate_fairness(y, predictions, A,
                                                        unprivileged_val=0, privileged_val=1)
    
    print("\\n=== Model Evaluation & Fairness Audit (Threshold 0.5) ===")
    print(f"Global Accuracy: {np.mean(predictions == y):.2%}")
    print(f"Equal Opportunity Difference (TPR Gap): {fairness_report['Equal_Opportunity_Difference']:.4f}")
    print("\\nDetailed Demographic Metrics:")
    print(f"  Group 0 -> TPR: {fairness_report['Unprivileged_Metrics']['TPR']:.4f}, Approval: {fairness_report['Unprivileged_Metrics']['P(Y_hat=1)']:.4f}")
    print(f"  Group 1 -> TPR: {fairness_report['Privileged_Metrics']['TPR']:.4f}, Approval: {fairness_report['Privileged_Metrics']['P(Y_hat=1)']:.4f}")
    print("===========================================================\\n")
    
    print("4. Simulating Threshold Calibration per Risk Band (Post-processing)...")
    print("   Lowering decision threshold for Group 0 to enforce Equal Opportunity.")
    
    # Search for an adjusted threshold for Group 0
    adjusted_threshold = 0.5
    probas = model.predict_proba(X_scaled)
    # Target TPR is the TPR of Group 1
    target_tpr = fairness_report['Privileged_Metrics']['TPR']
    
    for t_idx in np.arange(0.5, 0.2, -0.01):
        group_0_preds = (probas[A == 0] >= t_idx).astype(int)
        y_true_0 = y[A == 0]
        tp = np.sum((group_0_preds == 1) & (y_true_0 == 1))
        p = np.sum(y_true_0 == 1)
        current_tpr = tp / p if p else 0
        if current_tpr >= target_tpr:
            adjusted_threshold = t_idx
            break
            
    # Apply calibrated thresholds
    final_preds = np.zeros_like(y)
    final_preds[A == 1] = (probas[A == 1] >= 0.5).astype(int)
    final_preds[A == 0] = (probas[A == 0] >= adjusted_threshold).astype(int)
    
    calibrated_fairness = FairnessMetrics.evaluate_fairness(y, final_preds, A)
    print(f"   New threshold for Group 0: {adjusted_threshold:.3f}")
    print(f"   New Equal Opportunity Difference: {calibrated_fairness['Equal_Opportunity_Difference']:.4f}")
    print("   System structurally equalizes access for qualified applicants!")

if __name__ == "__main__":
    main()
