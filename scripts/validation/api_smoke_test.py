import requests

BASE_URL = "http://127.0.0.1:8000"

API_KEY = "geo-aware-mro-dev"

HEADERS = {
    "x-api-key": API_KEY
}

def validate_root():

    r = requests.get(f"{BASE_URL}/")

    print("ROOT:", r.status_code)

def validate_health():

    r = requests.get(f"{BASE_URL}/health")

    print("HEALTH:", r.status_code)

def validate_inference():

    payload = {
        "country": "India",
        "supplier_score": 0.82
    }

    r = requests.post(
        f"{BASE_URL}/inference/risk-score",
        json=payload,
        headers=HEADERS,
    )

    print("INFERENCE:", r.status_code)
    print(r.json())

def validate_distributed():

    r = requests.get(
        f"{BASE_URL}/distributed/status"
    )

    print("DISTRIBUTED:", r.status_code)
    print(r.json())

if __name__ == "__main__":

    validate_root()

    validate_health()

    validate_inference()

    validate_distributed()

    print("\nSmoke tests complete")
