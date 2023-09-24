import nltk, re, csv
import pandas as pd
from nltk.corpus import stopwords 

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
        if word in stopwords:   #Если слово в стоплисте, удаляем                
             del tokens[index]
        else:
            word = lemmatizer.lemmatize(word) #Лемматизация  
    return tokens

#Метод запуска предобработки файла с комментариями. Передаётся имя файла и bool-переменная для сохранения изменений в отдельном файле
def start_preprocessing(filename, save_file, save_words_list):
    data = pd.read_csv(filename) #Файл с кодом и датасет лежат в одной папке
    data_cleansed = [] #Список для хранения в нем обработанных комментариев
    for row in data["comment_text"]:    #каждый комментарий в датасете обрабатывается отдельным методом
        text = prepare_text(row)
        data_cleansed.append(str(text))
    data["comment_prepared"] = data_cleansed # Добавляем в датасет колонку для обработанного комментария с токенизацией на слова
    if save_file == True:
        data.to_csv('prepared_data.csv') #можно сохранить данные в датасете
    if save_words_list == True:
        with open ('word_tokens.csv', 'w', encoding='UTF8') as file:  #Можно сохранить отдельно очищенные комментарии
            fieldnames = ['id', 'words']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            n = 0
            for i in data_cleansed:
                writer.writerow({'id' : n, 'words' : i})
                n += 1

    return data

start_preprocessing("train.csv", False, True)


