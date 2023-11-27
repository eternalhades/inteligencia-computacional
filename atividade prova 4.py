import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Pontos de exemplo
X_train = np.array([[10, 20, 1], [8, 13, 0], [6, 15, 1], [5, 11, 1], [20, 26,0], [22, 26,0], [23, 27,1], [18, 23,0]])
y_train = np.array([1, 0, 1, 1, 0, 0, 1, 1])

# Pontos para classificar
c1 = np.array([5, 10])
c2 = np.array([25, 20])

# Função para calcular distância euclidiana
def euclidean_distance(p1, p2):
    return euclidean(p1, p2)

# Função para encontrar a classe com base nas k menores distâncias
def knn_predict(X_train, y_train, x, k=3):
    distances = [euclidean_distance(x, x_train) for x_train in X_train]
    indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[indices]
    return np.bincount(k_nearest_labels).argmax()

# Classificar c1
class_c1 = knn_predict(X_train, y_train, c1, k=3)
print(f"A classe para c1 é: {class_c1}")

# Classificar c2
class_c2 = knn_predict(X_train, y_train, c2, k=3)
print(f"A classe para c2 é: {class_c2}")

# Plotar os pontos
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(c1[0], c1[1], marker='X', s=100, label='c1', edgecolors='k')
plt.scatter(c2[0], c2[1], marker='X', s=100, label='c2', edgecolors='k')
plt.xlabel('Temperatura')
plt.ylabel('Umidade')
plt.title('Pontos de Treinamento e Pontos para Classificar')
plt.legend()
plt.show()