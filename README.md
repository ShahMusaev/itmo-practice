# itmo-practice

Анализ тональности текстов
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


