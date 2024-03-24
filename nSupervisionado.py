import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Pré-processamento
# Coleta e Integração
iris = load_iris()

caracteristicas = iris.data

# Mineração
grupos = KMeans(n_clusters=3) # Usando Kmeans - Algoritmos de clusterização - para agrupar amostras que possuem caracteristicas semelhantes
grupos.fit(X=caracteristicas) # treinando dados
labels = grupos.labels_ # indice do grupo ao qual cada amostra pertence

# Pós-processamento
# Criando dusa figuras com dados treinados
fig = plt.figure()
fig.add_subplot(projection='3d')
plt.xlabel('Comprimento Sépala')
plt.ylabel('Largura Sépala')
plt.clabel('Comprimento Pétala')
plt.scatter(caracteristicas[:, 0], caracteristicas[:, 1], caracteristicas[:, 2], c=grupos.labels_, edgecolor='k')

target = iris.target
fig = plt.figure()
fig.add_subplot(projection='3d')
plt.xlabel('Comprimento Sépala')
plt.ylabel('Largura Sépala')
plt.clabel('Comprimento Pétala')
plt.scatter(caracteristicas[:, 0], caracteristicas[:, 1], caracteristicas[:, 2], c=target, edgecolor='k')

plt.show()
