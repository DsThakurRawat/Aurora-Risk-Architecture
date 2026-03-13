import numpy as np
import pandas as pd

class SyntheticDataSimulator:
    """
    Simulates realistic Alternative Credit Data for thin-file consumers.
    Attributes include Cash-flow metrics, Telecom behavior, and Payment stability.
    """

    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        self.n = n_samples
        np.random.seed(random_state)

    def generate(self) -> pd.DataFrame:
        """
        Generate the synthetic population.
        Unprivileged groups (A=0) face structural instability but have varying repayment capabilities.
        """
        
        # Protected Attribute A (0 = Unprivileged, 1 = Privileged)
        # Structural bias: Unprivileged are minority (~30%)
        A = np.random.choice([0, 1], p=[0.3, 0.7], size=self.n)
        
        # 1. Alternative Signals
        # Inflow Consistency (higher is more stable).
        inflow_consistency = np.where(A == 1, 
                                      np.random.normal(0.7, 0.15, self.n), 
                                      np.random.normal(0.5, 0.2, self.n))
        inflow_consistency = np.clip(inflow_consistency, 0, 1)
        
        # Savings Buffer (days coverage, proxy for resilience).
        savings_buffer = np.where(A == 1,
                                  np.random.exponential(scale=60.0, size=self.n),
                                  np.random.exponential(scale=30.0, size=self.n))
                                  
        # Phone Tenure (months).
        phone_tenure = np.random.lognormal(mean=2.5, sigma=1.0, size=self.n)
        
        # Utility Timeliness (percentage of bills paid on time).
        utility_timeliness = np.where(A == 1,
                                      np.random.normal(0.85, 0.1, self.n),
                                      np.random.normal(0.75, 0.15, self.n))
        utility_timeliness = np.clip(utility_timeliness, 0, 1)

        # 2. Correlated Target Variable: Y (1 = Repay, 0 = Default)
        # Latent score is based heavily on objective cash flow & resilience.
        # This allows TPR parity checks because we know the "true" capability.
        latent_score = (
            2.0 * inflow_consistency + 
            0.5 * (savings_buffer / 30) + 
            1.5 * utility_timeliness + 
            np.random.normal(0, 0.5, self.n)
        )
        
        # Base repayment rate depends slightly on group due to unmeasured systemic factors,
        # but mostly driven by financial behavior.
        probability_repay = 1 / (1 + np.exp(- (latent_score - 2.8)))
        
        Y = (np.random.rand(self.n) < probability_repay).astype(int)

        df = pd.DataFrame({
            "A": A,
            "inflow_consistency": inflow_consistency,
            "savings_buffer_days": savings_buffer,
            "phone_tenure_months": phone_tenure,
            "utility_timeliness": utility_timeliness,
            "Y": Y
        })
        
        return df
