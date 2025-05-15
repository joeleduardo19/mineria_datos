import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Cargar dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Verificar columnas
print(df.columns)

# Seleccionar características y etiquetas (ejemplo con 'popularity' y 'vote_average' para clasificar por 'original_language')
X = df[['popularity', 'vote_average']]
y = df['original_language']

# Filtrar solo las 3 lenguas más frecuentes para simplificar el modelo
top_languages = y.value_counts().nlargest(3).index
filtered_df = df[df['original_language'].isin(top_languages)]
X = filtered_df[['popularity', 'vote_average']]
y = filtered_df['original_language']

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predecir
y_pred = knn.predict(X_test_scaled)

# Evaluación
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=top_languages, yticklabels=top_languages)
plt.title("Matriz de Confusión - KNN")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()

# Guardar la gráfica
output_dir = "Practica6"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "knn_confusion_matrix.png"))
plt.show()
