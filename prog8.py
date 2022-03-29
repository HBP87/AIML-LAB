from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
targets = iris.target_names
print("Class : number")
for i in range(len(targets)):
    print(targets[i], ':', i)

X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"])
kn = KNeighborsClassifier(1)
kn.fit(X_train, y_train)

for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)
    print("Actual:[{0}] [{1}],Predicted:{2} {3}".format(y_test[i], targets[y_test[i]], prediction, targets[prediction]))
print("\nAccuracy:", kn.score(X_test, y_test))

