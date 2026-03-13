# Aurora Risk Architecture - Fair Credit Scoring for the Unbanked

This project implements a fair, explainable, and accountable credit scoring system aimed at the unbanked and thin-file borrowers, as per the blueprint.

## Design Philosophy

- **Objective Shift**: Optimize for `Expected Default Loss s.t. Fairness + Access Constraints` rather than pure profit maximization.
- **Alternative Data**: Leverages non-traditional signals (cash-flow, mobile behavior, network topology) as a proxy for financial resilience.
- **Fairness by Design**: Encodes Equal Opportunity parity directly into model training and post-processing evaluation.
- **Explainability**: Prioritizes simple, transparent models (logistic regression, trees) and SHAP interpretations.
- **Zero Scikit-Learn Dependency**: Hand-implemented algorithms utilizing core `NumPy` and `PyTorch` libraries to give fine-grained control over algorithmic logic and bias constraints.

## Project Structure

```bash
.
├── src/
│   ├── data_pipeline/         # Data ingestion, synthetic simulation, and ETL
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineer.py
│   ├── evaluation/            # Predictive metrics and reporting
│   ├── explainability/        # Custom SHAP-like calculations and local explanations
│   ├── fairness/              # Bias auditing, pre-processing, and constrained training
│   │   ├── __init__.py
│   │   ├── fairness_metrics.py
│   │   └── fair_trainer.py
│   ├── models/                # Hand-coded ML algorithms (no sklearn)
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── logistic_regression.py
│   └── monitoring/            # Data drift detection (e.g., PSI, KL-divergence)
├── tests/                     # Unit tests
├── Notebooks/                 # Jupyter Notebooks for EDA and prototype simulation
├── README.md
└── requirements.txt
```

## Getting Started

1. Set up a virtual environment (Python 3.10+ recommended)
2. Install dependencies: `pip install -r requirements.txt`

### Ethics and Governance

Credit is compressed trust. This repository views credit scoring not as an isolated mathematical exercise but as a socio-technical governance system. Continuous monitoring and explicit ethical tradeoffs are core requirements for using this codebase.
# Aurora-Risk-Architecture
