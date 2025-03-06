import pickle

import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

x = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)  # pierwszy element ze zbioru iris dataset from sklearn.dataset, # predykcja powinna zwrócić 0
y_pred = model.predict(x)
print(y_pred)