# itmo-practice

Анализ тональности текстов
==================
Для удобства работы выбрана среда разработки PyCharm и подключены нужные модули. Код запускается с jupeter notebook, поэтому рекомендуется проверить наличие данного модуля

Данные для обучения
-----------------------------------
Данные для обучения были скачены с [корпуса твитов](http://study.mokoron.com/),сформированные на основе русскоязычных сообщений из Twitter .Он содержит 114 991 положительных, 111 923 отрицательных твитов, а также базу неразмеченных твитов объемом 17 639 674 сообщений.

Нужно скать файл db.sql  и преобразовать его в 

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
