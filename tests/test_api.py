from fastapi.testclient import TestClient
from src.serving.api import app
from src.serving.schemas import LoanApplication

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_score_application_success():
    # Valid payload matching the unbanked schema
    valid_payload = {
        "borrower_id": "USER_12345",
        "inflow_consistency": 0.85,
        "savings_buffer_days": 45.0,
        "phone_tenure_months": 24.5,
        "utility_timeliness": 0.95,
        "requested_amount": 500.0
    }
    
    response = client.post("/v1/score", json=valid_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "approved" in data
    assert "risk_score" in data
    assert "explanations" in data
    assert data["borrower_id"] == "USER_12345"

def test_score_application_validation_error():
    # Invalid payload (missing required field 'phone_tenure_months')
    invalid_payload = {
        "borrower_id": "USER_99999",
        "inflow_consistency": 0.5,
        "savings_buffer_days": 10.0,
        "utility_timeliness": 0.5,
        "requested_amount": 1000.0
    }
    
    response = client.post("/v1/score", json=invalid_payload)
    assert response.status_code == 422 # Pydantic Validation Error
