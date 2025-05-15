import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Carga de datos
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Verificar columnas disponibles
print("Columnas disponibles:", df.columns.tolist())

# Seleccionar las columnas numéricas que sí existen y usaremos para clustering
features = ['popularity', 'vote_average', 'vote_count']  

# Eliminar filas con datos faltantes en esas columnas
df_clean = df[features].dropna()

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Encontrar número óptimo de clusters con método del codo
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(2,11), wcss, marker='o')
plt.title('Método del codo para número óptimo de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS (Inercia)')
plt.savefig('elbow_method.png')
plt.show()

# Elegimos k=4 (por ejemplo, basado en el gráfico)
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Calcular métricas de calidad
silhouette = silhouette_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)
ch_score = calinski_harabasz_score(X_scaled, labels)

print(f'Silhouette Score: {silhouette:.3f}')
print(f'Davies-Bouldin Index: {db_index:.3f}')
print(f'Calinski-Harabasz Score: {ch_score:.3f}')

# Reducir dimensiones para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for cluster in range(k_optimo):
    plt.scatter(X_pca[labels == cluster, 0], X_pca[labels == cluster, 1], label=f'Cluster {cluster}')
plt.title('Clusters de películas con K-means (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.savefig('clusters_pca.png')
plt.show()
