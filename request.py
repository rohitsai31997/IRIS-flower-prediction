import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Sepal_Length': 6.7, 'Sepal_Width': 3.1, 'Petal_Length': 4.4 , 'Petal_Width': 1.4 })

print(r.json())