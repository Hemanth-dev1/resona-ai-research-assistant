import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8080", timeout=5) as response:
        print(f"Status: {response.status}")
except Exception as e:
    print(f"Error: {e}")
