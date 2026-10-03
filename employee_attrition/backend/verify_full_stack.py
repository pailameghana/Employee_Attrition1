import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000"

def test_full_stack():
    print("--- 1. Testing Health Endpoint ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/health") as response:
        data = json.loads(response.read().decode())
        print("Health response:", data)
        assert data["status"] == "online"

    print("\n--- 2. Testing Companies Endpoint ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/companies") as response:
        comp_data = json.loads(response.read().decode())
        print("Companies supported:", comp_data["companies"])
        print(f"Company summary count: {len(comp_data['summary'])}")
        assert "TCS" in comp_data["companies"]
        assert "Wipro" in comp_data["companies"]
        assert "Infosys" in comp_data["companies"]

    print("\n--- 3. Testing Streamlined Single Predictor API ---")
    sample_payload = {
        "Age": 32,
        "JobRole": "Sales Executive",
        "MonthlyIncome": 3200,
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 2,
        "DistanceFromHome": 21
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/predict/single",
        data=json.dumps(sample_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode())
        print(f"Prediction: Probability = {res['attrition_probability']} | Risk Band = {res['risk_band']}")
        print(f"Top Push Factor (Plain): {res['top_push_factors'][0]['plain']}")
        print(f"Actionable HR Recommendations Count: {len(res['recommendations'])}")

    print("\n--- 4. Testing Global Insights for TCS Dataset ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/insights/global?company=TCS") as response:
        insights = json.loads(response.read().decode())
        print("Risk distribution for TCS:", insights["risk_distribution"])
        print("Global top feature count:", len(insights["global_top_features"]))

    print("\n--- 5. Testing Model Performance Benchmark Report ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/model/metrics") as response:
        metrics = json.loads(response.read().decode())
        print(f"Metrics records count: {len(metrics)}")

    print("\n=======================================================")
    print("SUCCESS: FULL STACK APPLICATION VERIFIED END-TO-END!")
    print("=======================================================")

if __name__ == "__main__":
    test_full_stack()
