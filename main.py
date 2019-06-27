
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re


# Считываем данные
n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

# Формируем сбалансированный датасет
sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


data = [preprocess_text(t) for t in raw_data]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

import sqlite3

# Открываем SQLite базу данных
conn = sqlite3.connect('positive.db')
c = conn.cursor()

with open('tweets.txt', 'w', encoding='utf-8') as f:
    # Считываем тексты твитов
    for row in c.execute('SELECT ttext FROM *'):
        if row[0]:
            tweet = preprocess_text(row[0])
            # Записываем предобработанные твиты в файл
            print(tweet, file=f)