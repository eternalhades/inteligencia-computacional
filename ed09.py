from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data (assuming you already have the CSV file)
# Replace 'your_file.csv' with the actual file name.
arquivo_dados = 'your_file.csv'
dados_vinhos = pd.read_csv(arquivo_dados)

# Check for missing values
if dados_vinhos.isnull().any().any():
    print("Missing values detected.")
    # Include code for handling missing values if needed.

# Standardize the features
scaler = StandardScaler()
atributos_normalizados = scaler.fit_transform(dados_vinhos.drop('quality', axis=1))

# Cluster using Agglomerative Clustering
modelo_agglomerative = AgglomerativeClustering(n_clusters=3)
clusters_identificados = modelo_agglomerative.fit_predict(atributos_normalizados)

# Add Cluster Labels to the Dataset
dados_vinhos['rotulo_cluster'] = clusters_identificados

# Evaluate with Silhouette Score
silhouette_score_value = silhouette_score(atributos_normalizados, clusters_identificados)
print(f"Silhouette Score: {silhouette_score_value:.2f}")

# Visualize the Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='feature1', y='feature2', hue='rotulo_cluster', data=dados_vinhos, palette='viridis', s=100)
plt.title('Clusters by Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()