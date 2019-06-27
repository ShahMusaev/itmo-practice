# itmo-practice

Определние категории предложения
==================
Для удобства работы выбрана среда разработки PyCharm и подключены нужные модули. Код запускается с jupeter notebook, поэтому рекомендуется проверить наличие данного модуля

Данные для обучения
-----------------------------------
Данные для обучения были скачены с [корпуса твитов](http://study.mokoron.com/),сформированные на основе русскоязычных сообщений из Twitter .Он содержит 114 991 положительных, 111 923 отрицательных твитов, а также базу неразмеченных твитов объемом 17 639 674 сообщений.

Нужно загрузить файл db.sql  и преобразовать его для удовбства в sqlite. Это можно сделать с помощью специального [скрипта](https://github.com/dumblob/mysql2sqlite) (если у Вас Windows, следует установить Cygwim)

```python
import pandas as pd
import numpy as np

# Считываем данные
n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('data/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('data/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

# Формируем сбалансированный датасет
sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size```
```
Перед тем как начать обучение следует провести предварительную обработку данных  и разделить наши данные на обучающую и тестовую выборку
```python
import re

def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


data = [preprocess_text(t) for t in raw_data]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
```

Word2Vec - векторное представление слов
-----------------------------------
Для лучшего определния темы рекамендуется ознакомиться со статьей про [Word2Vec](https://habr.com/ru/post/249215/)

Теперь с помошью  библиотеки sqlite3 вытаскиваем нужную нам информацию 

```python
import sqlite3

# Открываем SQLite базу данных
conn = sqlite3.connect('twitter.db')
c = conn.cursor()
with open('tweets.txt', 'w', encoding='utf-8') as f:
    # Считываем тексты твитов 
    count = 300001
    for row in c.execute('SELECT ttext FROM sentiment'):
        if row[0]:
            tweet = preprocess_text(row[0])
            # Записываем предобработанные твиты в файл
            print(tweet, file=f)
            count = count-1
            if count < 0:
                break
```
Далее с помощью библиотеки Gensim обучаем нашу  Word2Vec-модель и присваиваем  следующие параметры: 
(size = 200 — размерность признакового пространства;
window = 5 — количество слов из контекста, которое анализирует алгоритм;
min_count = 3 — слово должно встречаться минимум три раза, чтобы модель его учитывала.)

```python
import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Считываем файл с предобработанными твитами

data = gensim.models.word2vec.LineSentence('tweets.txt')

# Обучаем модель 
model = Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())
model.save("model.w2v")
```

Визуализация векторного представления текстов
-----------------------------------
Для наглядности возьмем категории: 'погода', 'еда','игры','экономика','фильм','одежда' и с помощью алгоритма визуализации t-SNE наблюдаем картинку, где схожие слова расположены близко 


```python
keys = ['погода', 'еда','игры','экономика','фильм','одежда']
embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=40):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words))
    
tsne_model_en_2d = TSNE(perplexity=10, n_components=5, init='pca', n_iter=3500, random_state=32)
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# % matplotlib inline


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()


tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)
```

