from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = datasets.load_iris()
features = iris_dataset.data
labels = iris_dataset.target

print(features[56], labels[52])

'''
print(iris_dataset.DESCR)
class:
    - Iris-Setosa
    - Iris-Versicolour
    - Iris-Virginica
'''

classifier = KNeighborsClassifier()

classifier.fit(features, labels)
prediction = classifier.predict([features[56]])
print(prediction)

if prediction == [1]:
    print('Its Setosa')
elif prediction == [2]:
    print('Its Versicolour')
else:
    print('Its Virginica')


