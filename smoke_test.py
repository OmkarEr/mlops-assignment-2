import requests
import sys

try:
    # Test Health Endpoint
    resp = requests.get("http://localhost:8000/health")
    resp.raise_for_status()
    print("Smoke Test Passed: Health check OK.")
    sys.exit(0)
except Exception as e:
    print(f"Smoke Test Failed: {e}")
    sys.exit(1)