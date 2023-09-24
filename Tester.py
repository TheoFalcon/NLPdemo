import requests
url = 'http://localhost:5000/api'
text = input()
r = requests.post(url,json={'text' : f'{text}'})
print(r.json())