"""
    Train a Logistic Regression classifier to predict whether the flower is Iris Virginica or not
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

get_data = datasets.load_iris()
print(get_data.keys())
"""
    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""
x = get_data['data'][:, 3:]
y = (get_data['target'] == 2).astype(np.int)

"""
    Train a logistic Regression Classifier
"""
classifier = LogisticRegression()
classifier.fit(x, y)
output = classifier.predict(([[2.6]])) # Change this Value to get Output
print(output)


"""
    Graphical Representation using Matplotlib
"""

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_probability = classifier.predict_proba(x_new)
plt.plot(x_new, y_probability[:, 1], label='virgenica')
plt.legend()
plt.grid()
plt.show()