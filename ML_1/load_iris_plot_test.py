from sklearn.datasets import load_iris

iris = load_iris()
import matplotlib.pyplot as plt

plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)

markers = ['o', '^', 'v']
for i in range(3):
    d = iris.data[iris.target == i, :]
    plt.plot(d[:,0], d[:,1], 'o', fillstyle='none', marker=markers[i])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(iris.target_names)
from sklearn.decomposition import PCA

x = PCA(n_components=2).fit_transform(iris.data)

for i in range(3):
    d = x[iris.target == i, :]
    plt.plot(d[:,0], d[:,1], 'o', fillstyle='none', marker=markers[i])
plt.xlabel('1st')
plt.ylabel('2nd')
plt.legend(iris.target_names)