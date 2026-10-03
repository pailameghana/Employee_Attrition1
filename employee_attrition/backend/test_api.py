from dataset_manager import get_cleaned_dataset
from shap_engine import SHAPEngine
from main import get_model_metrics

def test_backend_direct():
    print("1. Testing Dataset Manager...")
    df = get_cleaned_dataset()
    print(f"Dataset shape: {df.shape}")
    assert df.shape[0] > 0

    print("\n2. Testing SHAP Engine...")
    engine = SHAPEngine()
    sample_input = {
        "Age": 35,
        "JobRole": "Sales Executive",
        "MonthlyIncome": 5813,
        "OverTime": "Yes",
        "JobSatisfaction": 4,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 10,
        "DistanceFromHome": 27
    }
    res = engine.predict_single(sample_input)
    print(f"Prediction result: Probability={res['attrition_probability']}, Risk Band={res['risk_band']}")
    assert "attrition_probability" in res
    assert "top_push_factors" in res
    assert len(res["recommendations"]) > 0

    print("\n3. Testing Model Metrics Endpoint...")
    metrics = get_model_metrics()
    model_names = [m.get("Model") for m in metrics]
    print("Evaluated Models:", model_names)
    assert any("Stacking Ensemble" in name for name in model_names)

    print("\nALL BACKEND LOGIC VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_backend_direct()

