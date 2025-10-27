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
def carregar():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['specie'] = iris.target_names[iris.target]
    df['target'] = iris.target
    return df,iris

# Função que define quais colunas vão ser separadas para a análise dos dados.

def metric(df,x,y):
    return df.loc[df.target.isin([0,1,2]),[x, y,'target']]

def train(df):
# Criação das variáveis X e y. X são os dados e y a espécie.

    X = df.drop('target',axis=1)
    y = df.target

# O split test separa alguns dados para treinar e outros para testar e validar a árvore.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)

# Criação da árvore de decisão utilizando o critério entropia com os dados de treino.

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train,y_train)

# Os valores separados para teste são classificados e seus resultados são conferidos com o y_test.

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    print ( f"Accuracy: {accuracy} " )
    print (y_test, "\n" ,y_pred)
    print(X_test)
    return X_train,X_test,y_train,y_test,clf

# Plot do gráfico e da árvore.

def plot(X,y,clf):
    ax = plt.subplot(1,2,1)

    ax.scatter(X['sepal length (cm)'],
        X['sepal width (cm)'], 
        c=y #roxo: setosa, verde: versicolor, amarelo: virginica
        )
    ax.set(
        xlim=(min(X['sepal length (cm)']),max(X['sepal length (cm)'])),
        ylim=(min(X['sepal width (cm)']),max(X['sepal width (cm)']))
        )

    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('sepal width (cm)')

    plt.subplot(1,2,2)
    tree.plot_tree(clf,fontsize=5)

    plt.show()

# Função 

def classify(X,clf,iris):
    return iris.target_names[clf.predict(X)]


df,iris = carregar()
df = metric(df,'sepal length (cm)','sepal width (cm)')
X_train,X_test,y_train,y_test,clf = train(df)

print(classify([[6,2.5]],clf,iris)[0])


plot(X_train,y_train,clf)