from abc import ABC, abstractmethod
import numpy as np

class CreditModel(ABC):
    """
    Abstract Base Class for all transparent and complex credit scoring models.
    Adheres to rigorous Object-Oriented Design to ensure standard contracts
    across different architectures.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> None:
        """
        Train the model on the provided tabular dataset.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Binary labels (0 = Default, 1 = Repayment).
            sample_weights (np.ndarray): Optional weights for balanced training or fair reweighing.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict repayment probabilities.
        
        Args:
            X (np.ndarray): Feature matrix.
            
        Returns:
            np.ndarray: Probabilities representing P(Y=1|X)
        """
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels based on a threshold.
        
        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Decision threshold (can be adjusted per group for fairness).
            
        Returns:
            np.ndarray: Predicted binary classes.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Basic evaluation of accuracy and baseline AUC.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.
            
        Returns:
            dict: Baseline performance metrics (Fairness is handled globally by fairness modules).
        """
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return {"accuracy": accuracy}
