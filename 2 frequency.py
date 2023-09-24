import nltk
from nltk import FreqDist
import pandas as pd


fd = FreqDist()
data = pd.read_csv("word_tokens.csv", encoding='utf-8') #открываем файл в папке со скриптом

for row in data['words']:
    row_formatted = row[1:-1]  # удаляем символы [] в начале и конце строки 
    words = nltk.word_tokenize(row_formatted, language = 'english')
    for word in words:
       if (word == '') or (word == "'"):
           pass
       elif len(word) <4: #я решил добавить условие на длину слов, чтобы не отображать слова типа 'a', 'be' и другие
           pass
       else:
            word_f = word[1:-1] # удаляем символы '' в начале и конце слова 
            fd[word_f] += 1
fd.plot(10)


