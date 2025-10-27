import pandas as pd
import numpy as np
import graphviz 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.datasets import load_iris

# Carrega e formata os dados
'''
index/key | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | specie | target

target[0,1,2], onde 0 é setosa, 1 é versicolor e 2 é virginica.
'''
 
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['specie'] = iris.target_names[iris.target]
df['target'] = iris.target

# Função que define quais colunas vão ser separadas para a análise dos dados.

def metric(x,y):
    return df.loc[df.target.isin([0,1,2]),[x, y,'target']]

df1 = metric('petal length (cm)','petal width (cm)')

# Criação das variáveis X e y. X são os dados e y a espécie.

X = df1.drop('target',axis=1)
y = df1.target

# O split test separa alguns dados para treinar e outros para testar e validar a árvore.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Criação da árvore de decisão utilizando o critério entropia com os dados de treino.

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)

# Os valores separados para teste são classificados e seus resultados são conferidos com o y_test.

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print ( f"Accuracy: {accuracy} " )
print (y_test, "\n" ,y_pred)

# Plot do gráfico e da árvore.

ax = plt.subplot(1,2,1)

ax.scatter(X_test['petal length (cm)'],
    X_test['petal width (cm)'], 
    c=y_test)
ax.set(xlim=(0,8),xticks=np.arange(1,8),
    ylim=(0,8),yticks=np.arange(1,8))

plt.subplot(1,2,2)
tree.plot_tree(clf,fontsize=7)

plt.show()