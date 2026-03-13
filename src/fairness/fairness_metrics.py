import numpy as np

class FairnessMetrics:
    """
    Core component for evaluating algorithmic fairness for credit scoring models.
    Provides rigorous calculations for Equal Opportunity, Demographic Parity, etc.
    """
    
    @staticmethod
    def _compute_rates(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray, group_val: int) -> dict:
        """Helper function to compute TPR, FPR, FNR, P(Y^=1) for a specific group."""
        group_mask = (A == group_val)
        y_t = y_true[group_mask]
        y_p = y_pred[group_mask]
        
        # True Positives and Total Actual Positives
        tp = np.sum((y_p == 1) & (y_t == 1))
        p = np.sum(y_t == 1)
        
        # True Negatives and Total Actual Negatives
        tn = np.sum((y_p == 0) & (y_t == 0))
        n = np.sum(y_t == 0)
        
        tpr = tp / p if p > 0 else 0
        fpr = (np.sum(y_p == 1) - tp) / n if n > 0 else 0
        approval_rate = np.mean(y_p == 1) if len(y_p) > 0 else 0
        
        return {"TPR": tpr, "FPR": fpr, "P(Y_hat=1)": approval_rate, "Total": len(y_t)}

    @classmethod
    def evaluate_fairness(cls, y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray, 
                        unprivileged_val: int = 0, privileged_val: int = 1) -> dict:
        """
        Computes the primary fairness metrics:
        1. Equal Opportunity Difference (EOD): TPR difference
        2. Statistical Parity Difference (SPD): Approval rate difference
        3. Disparate Impact (DI): Ratio of approval rates
        """
        
        metrics_unprivileged = cls._compute_rates(y_true, y_pred, A, unprivileged_val)
        metrics_privileged = cls._compute_rates(y_true, y_pred, A, privileged_val)
        
        eod = metrics_unprivileged["TPR"] - metrics_privileged["TPR"]
        spd = metrics_unprivileged["P(Y_hat=1)"] - metrics_privileged["P(Y_hat=1)"]
        
        # Avoid division by zero
        di = metrics_unprivileged["P(Y_hat=1)"] / metrics_privileged["P(Y_hat=1)"] if metrics_privileged["P(Y_hat=1)"] > 0 else 1.0
        
        return {
            "Equal_Opportunity_Difference": eod,
            "Statistical_Parity_Difference": spd,
            "Disparate_Impact_Ratio": di,
            "Unprivileged_Metrics": metrics_unprivileged,
            "Privileged_Metrics": metrics_privileged
        }
