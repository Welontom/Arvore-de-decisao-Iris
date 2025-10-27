from avl_tree_iris import AvlTreeIris
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy.stats import norm

# Carregar o conjunto Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Funcao para calcular indice composto
def calculate_composite_index(row):
    return np.mean(row[iris.feature_names])

# Criar arvores AVL para cada especie
avl_setosa = AvlTreeIris()
avl_versicolor = AvlTreeIris()
avl_virginica = AvlTreeIris()

# Dicionario para mapear especies as arvores
species_trees = {
'setosa': avl_setosa,
'versicolor': avl_versicolor,
'virginica': avl_virginica
}

# Inserir dados nas arvores AVL
for index, row in df.iterrows():
    species = row['species']
    composite_index = calculate_composite_index(row)
    species_trees[species].insert(composite_index, index)

# Calcular intervalos de confianca (95%)
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ci_lower, ci_upper = norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(data)))
    return mean, ci_lower, ci_upper

# Exemplo: Intervalo de confianca para comprimento da petala
for species in df['species'].unique():
    species_data = df[df['species'] == species]['petal length (cm)']
    mean, ci_lower, ci_upper = calculate_confidence_interval(species_data)
    print(f"{species}: Media = {mean:.2f}, Intervalo de Confianca (95%) = [{ci_lower:.2f}, {ci_upper:.2f}]")

# Funcao de classificacao
def classify_sample(sample):
    composite_index = calculate_composite_index(sample)
    min_diff = float('inf')
    predicted_species = None
    for species, tree in species_trees.items():
        closest_value = tree.find_closest(composite_index)
        if closest_value and abs(closest_value - composite_index) < min_diff:
            min_diff = abs(closest_value - composite_index)
            predicted_species = species
    return predicted_species

# Exemplo de uso
sample = pd.Series([5.1, 3.5, 1.4, 0.2], index=iris.feature_names)
predicted = classify_sample(sample)
print(f"Amostra classificada como: {predicted}")

# Relatorio da estrutura da arvore
for species, tree in species_trees.items():
    print(f"Arvore para {species}: Altura = {tree.height()}, Nos = {tree.size()}")

