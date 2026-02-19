import pandas as pd
import numpy as np
import graphviz 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.datasets import load_iris

# Carrega e formata os dados
'''
index/key | petal length (cm) | petal width (cm) | sepal length (cm) | sepal width (cm) | specie | target

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Criação da árvore de decisão utilizando o critério entropia com os dados de treino.

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train,y_train)

# Os valores separados para teste são classificados e seus resultados são conferidos com o y_test.

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    print ( f"Accuracy: {accuracy} " )
    return X_train,X_test,y_train,y_test,clf

# Plot do gráfico e da árvore.

def plot(X,y,clf):
    ax = plt.subplot(1,2,1)

    ax.scatter(X['petal length (cm)'],
        X['petal width (cm)'], 
        c=y #roxo: setosa, verde: versicolor, amarelo: virginica
        )
    ax.set(
        xlim=(min(X['petal length (cm)']) -0.5,max(X['petal length (cm)'])+0.5),
        ylim=(min(X['petal width (cm)']) -0.5,max(X['petal width (cm)'])+0.5)
        )

    ax.set_xlabel('petal length (cm)')
    ax.set_ylabel('petal width (cm)')

    plt.subplot(1,2,2)
    tree.plot_tree(clf,fontsize=7, impurity=True, filled=True)

    plt.show()

def plot_all(sdf):
    for i in range(0,len(sdf)):
        ax = plt.subplot(3,2,i+1)

        ax.scatter(sdf[i][0][sdf[i][1]],
            sdf[i][0][sdf[i][2]], 
            c=sdf[i][0].target #roxo: setosa, verde: versicolor, amarelo: virginica
            )
        ax.set(
            xlim=(min(sdf[i][0][sdf[i][1]]) -0.5,max(sdf[i][0][sdf[i][1]])+0.5),
            ylim=(min(sdf[i][0][sdf[i][2]]) -0.5,max(sdf[i][0][sdf[i][2]])+0.5)
            )

        ax.set_xlabel(sdf[i][1])
        ax.set_ylabel(sdf[i][2])
    plt.show()

# Função para classificar uma

def classify(X,clf,iris):
    return iris.target_names[clf.predict(X)]

def teste():
    df,iris = carregar()

    df1 = metric(df,'petal length (cm)','petal width (cm)')
    X_train1,X_test1,y_train1,y_test1,clf1 = train(df1)

    df2 = metric(df,'petal length (cm)','sepal length (cm)')
    X_train2,X_test2,y_train2,y_test2,clf2 = train(df2)

    df3 = metric(df,'petal length (cm)','sepal width (cm)')
    X_train3,X_test3,y_train3,y_test3,clf3 = train(df3)

    df4 = metric(df,'petal width (cm)','sepal length (cm)')
    X_train4,X_test4,y_train4,y_test4,clf4 = train(df4)

    df5 = metric(df,'petal width (cm)','sepal width (cm)')
    X_train5,X_test5,y_train5,y_test1,clf5 = train(df5)

    df6 = metric(df,'sepal length (cm)','sepal width (cm)')
    X_train5,X_test5,y_train5,y_test5,clf5 = train(df6)
    
    sdf = [ [df1, 'petal length (cm)','petal width (cm)'] ,
            [df2 ,'petal length (cm)','sepal length (cm)'],
            [df3 ,'petal length (cm)','sepal width (cm)'],
            [df4 ,'petal width (cm)','sepal length (cm)'],
            [df5 ,'petal width (cm)','sepal width (cm)'],
            [df6 ,'sepal length (cm)','sepal width (cm)']
    ]
    
    plot_all(sdf)

    plot(X_train1,y_train1,clf1)


teste()