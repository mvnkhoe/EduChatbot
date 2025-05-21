import requests

url = "http://127.0.0.1:8000/chat"
data = {
    "message": "Any tip"
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
