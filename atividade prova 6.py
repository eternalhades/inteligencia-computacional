import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inicializar parâmetros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Aprender pesos
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Regra de atualização do Perceptron
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Criar dados de exemplo
X_train = np.array([[10, 20, 1], [8, 13, 0], [6, 15, 1], [5, 11, 1], [20, 26, 0], [22, 26, 0], [23, 27, 1], [18, 23, 0]])
y_train = X_train[:, -1]  # Última coluna como rótulo
X_train = X_train[:, :-1]  # Todas as colunas exceto a última

# Treinar o Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# Fazer previsões nos dados de treinamento
predictions_train = perceptron.predict(X_train)

# Avaliar a precisão do Perceptron nos dados de treinamento
accuracy_train = accuracy(y_train, predictions_train)
print("Perceptron classification accuracy on training data:", accuracy_train)

# Visualizar a execução do algoritmo passo a passo
fig, ax = plt.subplots(2, 1, figsize=(8, 12))

# Plotar os dados de treinamento
ax[0].scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
ax[0].set_title('Dados de Treinamento')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')

# Plotar a fronteira de decisão aprendida pelo Perceptron
x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])
x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]
ax[0].plot([x0_1, x0_2], [x1_1, x1_2], "k--", label="Fronteira de Decisão")
ax[0].legend()

# Plotar as previsões nos dados de treinamento
ax[1].scatter(X_train[:, 0], X_train[:, 1], marker="o", c=predictions_train)
ax[1].set_title('Previsões nos Dados de Treinamento')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')

plt.show()
