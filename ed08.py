import requests
from IPython.display import Image, display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Display the image
url = 'https://images.pexels.com/photos/10343578/pexels-photo-10343578.jpeg'
display(Image(requests.get(url).content, width=800))

# Gere dados de exemplo
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 amostras com 2 características cada
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Classe 1 se a soma das características for maior que 1, Classe 0 caso contrário

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialize o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=3)

# Treine o modelo
knn.fit(X_train, y_train)

# Faça previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Calcule a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")

# Calcula a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Exibe a matriz de confusão
print("Matriz de Confusão:")
print(conf_matrix)

# Cria um heatmap da matriz de confusão
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
