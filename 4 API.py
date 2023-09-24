import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer


model = pickle.load(open('demo_model.sav', 'rb')) #загружаем модель
app = Flask(__name__)

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)  #получаем данные из запроса
    data2 = data['text']
    data2 = [data2]
    cv = CountVectorizer()   
    prediction = model.predict([[cv.fit_transform(data2)]]) #делаем прогноз с помощью созданной модели   
    output = prediction[0] #Забираем первое значение прогноза 
    return jsonify(output)

if __name__ == '__main__':
   app.run(port=5000, debug=True)