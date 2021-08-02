#Train a logistic regression classifier to predict whether a flower is iris ,virginica or not
from sklearn  import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris=datasets.load_iris()
#print(iris["target"])
#print(iris["DESCR"])
#print(list(iris.keys()))
x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int32)
clf=LogisticRegression()
clf.fit(x,y)
ex=clf.predict([[1.6]])
print(ex)
#using matplotlib toplot the visualisation
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=clf.predict_proba(x_new)
plt.plot(x_new,y_prob[0:1],"-g","virginica")
plt.show()
#print(y)
#print(x)
