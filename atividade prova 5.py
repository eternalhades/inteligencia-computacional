import numpy as np
import matplotlib.pyplot as plt

def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:

    def __init__(self, K=5, max_iteracoes=100, exibir_etapas=False):
        self.K = K
        self.max_iteracoes = max_iteracoes
        self.exibir_etapas = exibir_etapas
        self.clusters = [[] for _ in range(self.K)]
        self.centroides = []

    def prever(self, X):
        self.X = X
        self.n_amostras, self.n_caracteristicas = X.shape

        # Inicialização
        indices_aleatorios = np.random.choice(self.n_amostras, self.K, replace=False)
        self.centroides = [self.X[idx] for idx in indices_aleatorios]

        # Otimização dos clusters
        for _ in range(self.max_iteracoes):
            # Atribui amostras aos centroides mais próximos (cria clusters)
            self.clusters = self._criar_clusters(self.centroides)

            if self.exibir_etapas:
                self.plotar()

            # Calcula novos centroides a partir dos clusters
            centroides_antigos = self.centroides
            self.centroides = self._obter_centroides(self.clusters)

            if self._convergiu(centroides_antigos, self.centroides):
                break

            if self.exibir_etapas:
                self.plotar()

        # Classifica as amostras como o índice de seus clusters
        return self._obter_rotulos_clusters(self.clusters)

    def _obter_rotulos_clusters(self, clusters):
        # Cada amostra receberá o rótulo do cluster ao qual foi atribuída
        rotulos = np.empty(self.n_amostras)
        for indice_cluster, cluster in enumerate(clusters):
            for indice_amostra in cluster:
                rotulos[indice_amostra] = indice_cluster
        return rotulos

    def _criar_clusters(self, centroides):
        # Atribui as amostras aos centroides mais próximos
        clusters = [[] for _ in range(self.K)]
        for indice, amostra in enumerate(self.X):
            indice_centroide = self._centroide_mais_proximo(amostra, centroides)
            clusters[indice_centroide].append(indice)
        return clusters

    def _centroide_mais_proximo(self, amostra, centroides):
        # Distância da amostra atual para cada centroide
        distancias = [distancia_euclidiana(amostra, ponto) for ponto in centroides]
        indice_mais_proximo = np.argmin(distancias)
        return indice_mais_proximo

    def _obter_centroides(self, clusters):
        # Atribui o valor médio dos clusters aos centroides
        centroides = np.zeros((self.K, self.n_caracteristicas))
        for indice_cluster, cluster in enumerate(clusters):
            media_cluster = np.mean(self.X[cluster], axis=0)
            centroides[indice_cluster] = media_cluster
        return centroides

    def _convergiu(self, centroides_antigos, centroides):
        # Distâncias entre centroides antigos e novos, para todos os centroides
        distancias = [distancia_euclidiana(centroides_antigos[i], centroides[i]) for i in range(self.K)]
        return sum(distancias) == 0

    def plotar(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, indice in enumerate(self.clusters):
            ponto = self.X[indice].T
            ax.scatter(*ponto)

        for ponto in self.centroides:
            ax.scatter(*ponto, marker="x", color="black", linewidth=2)

        plt.show()


# Teste com os dados fornecidos
X_treino = np.array([[10, 20], [8, 13], [6, 15], [5, 11], [20, 26], [22, 26], [23, 27], [18, 23]])
y_treino = np.array([1, 0, 1, 1, 0, 0, 1, 1])

k = KMeans(K=2, max_iteracoes=150, exibir_etapas=True)
y_pred = k.prever(X_treino)

k.plotar()
