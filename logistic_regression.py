# train a logistic regression classifier to predict whether a flower is iris verginica or not
from sklearn import datasets
from sklearn.linear_model import  LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()

# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])

x = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int16)

# train aa logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[2.6]]))
print(example)

# using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new, Y_prob[:,1], "g-")
plt.show()

# print(y)
# print(x)

