import numpy as np
from src.models.base_model import CreditModel

class LogisticRegressionModel(CreditModel):
    """
    Hand-coded Logistic Regression optimized via Gradient Descent.
    
    Provides transparent coefficients suitable for Explainability
    (via SHAP or native weights analysis).
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, lambda_reg: float = 0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> None:
        """
        Trains the logistic regression using gradient descent.
        Supports sample weights for fairness reweighing (e.g., Kamiran & Calders).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        
        # Default weight array
        if sample_weights is None:
            sample_weights = np.ones(num_samples)

        for _ in range(self.epochs):
            # linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # predictions
            y_predicted = self._sigmoid(linear_model)
            
            # Gradients with sample weighting
            error = (y_predicted - y) * sample_weights
            
            # Incorporate L2 Regularization (ignoring bias term for simplicity here)
            dw = (1 / num_samples) * np.dot(X.T, error) + (self.lambda_reg / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(error)
            
            # Variable updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Returns P(Y=1|X) """
        if self.weights is None:
            raise ValueError("Model weights uninitialized. Please call fit() first.")
            
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
