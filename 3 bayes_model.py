from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import sklearn
import pickle


data = pd.read_csv("prepared_data.csv")   #получаем данные для обработки
data_rows_split = int((len(data.index))*0.2) #я беру только 20% выборки для обучения модели, т.к. памяти компьютера не хватает для большой матрицы
df_train = data.iloc[:data_rows_split] #данные для создания модели


count_vec = CountVectorizer()
bow = count_vec.fit_transform(df_train['comment_prepared']) #Использую Count Vectorizer для обработки текстовых данных
X= np.array(bow.todense())  #данные для оси Х
y = df_train['toxic'] #данные для оси У
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y) #разделяю данные на обучающие и тестовые
model = MultinomialNB().fit(X_train, y_train) #создаю модель

y_pred = model.predict(X_test) #проверяю точность модели на тестовой выборке и далее и вывожу результаты
print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred))
print('F1 score:', sklearn.metrics.f1_score(y_test, y_pred, average="macro"))
print(sklearn.metrics.classification_report(y_test, y_pred))

pickle.dump(model, open('demo_model.sav', 'wb')) #сохраняем модель


