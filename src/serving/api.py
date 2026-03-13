from fastapi import FastAPI, HTTPException, BackgroundTasks
import numpy as np
import time
import os
from src.serving.schemas import LoanApplication, ScoringDecision
# A mock abstract model for now (in production this would be loaded via MLflow)
from src.models.logistic_regression import LogisticRegressionModel

app = FastAPI(
    title="Aurora Risk Service",
    description="High-performance model serving for unbanked micro-credit.",
    version="1.0.0"
)

# Mocked state for the deployed model
def load_model():
    # In reality: model = mlflow.pyfunc.load_model("models:/AuroraFairLogReg/Production")
    model = LogisticRegressionModel()
    # Mocking trained weights
    model.weights = np.array([-0.5, -0.3, -0.1, -0.4]) 
    model.bias = 1.0
    return model

MODEL = load_model()

# Base logic for group thresholding (loaded from Redis/Config Map in prod)
THRESHOLDS = {
    "default_band": 0.5,
    "unprivileged_band": 0.42 # Example of calibrated threshold for Equal Opportunity
}

def feature_engineering(app_data: LoanApplication) -> np.ndarray:
    """
    Converts the incoming Pydantic payload into the required model tensor.
    In an enterprise system, this step might also pull historical aggregations
    from a Feature Store (like Feast) using the borrower_id.
    """
    return np.array([
        app_data.inflow_consistency,
        app_data.savings_buffer_days,
        app_data.phone_tenure_months,
        app_data.utility_timeliness
    ])
    
def log_decision_async(decision: ScoringDecision, broker_url: str = "localhost:9092"):
    """
    Background task to emit the scoring event to Kafka for auditing
    without blocking the API response latency.
    """
    # pseudo-code for Kafka emission
    # producer.send("aurora-decisions-log", value=decision.json())
    pass

@app.post("/v1/score", response_model=ScoringDecision)
async def score_application(application: LoanApplication, background_tasks: BackgroundTasks):
    """
    Real-time inference endpoint.
    """
    try:
        start_time = time.time()
        
        # 1. Prepare Features
        features = feature_engineering(application)
        
        # 2. Score
        probas = MODEL.predict_proba(features.reshape(1, -1))[0]
        
        # 3. Apply Governance Thresholds
        # In a real system, the user's demographic risk band might be inferred securely
        # or managed via an embedded demographic service.
        applied_threshold = THRESHOLDS["default_band"] 
        
        approved = bool(probas >= applied_threshold)
        
        # 4. Construct Response
        decision = ScoringDecision(
            borrower_id=application.borrower_id,
            approved=approved,
            risk_score=float(probas),
            risk_band="standard",
            applied_threshold=applied_threshold,
            explanations={
                # Mock explanations based on weights * input
                "inflow_consistency": float(features[0] * MODEL.weights[0]),
                "utility_timeliness": float(features[3] * MODEL.weights[3])
            }
        )
        
        # 5. Async Auditing
        broker_url = os.environ.get("KAFKA_BROKER_URL", "localhost:9092")
        background_tasks.add_task(log_decision_async, decision, broker_url=broker_url)
        
        return decision
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "v1-logistic"}
