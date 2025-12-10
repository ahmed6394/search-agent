import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("BRAVE_API")
url = "https://api.search.brave.com/res/v1/web/search"
params = {"q": "weather today in bretzfeld", "count": 3}
headers = {"Accept": "application/json", "X-Subscription-Token": token}

resp = requests.get(url, params=params, headers=headers, timeout=10)
# if resp.status_code == 200:
#     data = resp.json()
#     titles = [item["title"] for item in data.get("web", {}).get("results", [])]
#     print("✓ Brave API works. Titles:", titles)
# else:
#     print("✗ Error", resp.status_code, resp.text)

print(resp.text)