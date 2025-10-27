import pandas as pd
import numpy as np
import graphviz 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['specie'] = iris.target_names[iris.target]
df['target'] = iris.target

def metric(x,y):
    return df.loc[df.target.isin([0,1,2]),[x, y,'target']]

df1 = metric('petal length (cm)','petal width (cm)')

X = df1.drop('target',axis=1)
y = df1.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print ( f"Accuracy: {accuracy} " )
print (y_test, "\n" ,y_pred)

ax = plt.subplot(1,2,1)

ax.scatter(X_test['petal length (cm)'],
    X_test['petal width (cm)'], 
    c=y_test)
ax.set(xlim=(0,8),xticks=np.arange(1,8),
    ylim=(0,8),yticks=np.arange(1,8))

plt.subplot(1,2,2)
tree.plot_tree(clf,fontsize=7)

plt.show()