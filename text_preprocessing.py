import nltk, re
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.probability import FreqDist

stopwords = stopwords.words("english")
lemmatizer = nltk.stem.WordNetLemmatizer()


#Метод для предобработки текста
def prepare_text(text):    
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+]' #Задаём шаблоны регулярных выражений для пунктуации
    regular_digits = r'\d+' #Задаём шаблоны регулярных выражений для чисел
    regular_date = r'[\d{1,2}\/\d{1,2}\/\d{2,4}]' #Задаём шаблоны регулярных выражений для дат
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)' #Задаём шаблоны регулярных выражений для ссылок
    text = text.lower() #Переводим текст в нижний регистр и начинаем удалять пунктуацию, даты и пр:
    text = re.sub(regular, '', text)
    text = re.sub(regular_date, 'DATE', text)
    text = re.sub(regular_url, 'URL', text)
    text = re.sub(regular_digits, 'NUM', text)
    #Проверка на стопслова:
    tokens = nltk.word_tokenize(text) #Разбиваем комментарий на слова
    for word in tokens:
        index = tokens.index(word)
        word = lemmatizer.lemmatize(word) #Лемматизация        
        if word in stopwords:   #Если слово в стоплисте, удаляем                
             del tokens[index]
    return tokens

#Метод запуска предобработки файла с комментариями. Передаётся имя файла и bool-переменная для сохранения изменений в отдельном файле
def start_preprocessing(filename, save_file):
    data = pd.read_csv(filename) #Файл с кодом и датасет лежат в одной папке
    data_cleansed = []
    for row in data["comment_text"]:    
        text = prepare_text(row)
        data_cleansed.append(str(text))
    data["comment_prepared"] = data_cleansed # Добавляем в датасет колонку для обработанного комментария с токенизацией на слова
    if save_file == True:
        data.to_csv('prepared_data.csv')
    return data

start_preprocessing("train.csv", False)


