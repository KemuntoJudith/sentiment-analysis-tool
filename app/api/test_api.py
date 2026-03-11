# Test the API endpoint for aspect-based sentiment analysis
# This script sends a POST request to the API endpoint with a sample text and prints the response

import requests

url = "http://127.0.0.1:5000/predict-sentiment"

data = {
    "text": "The mobile banking app is very slow"
}

response = requests.post(url, json=data)

print(response.json())