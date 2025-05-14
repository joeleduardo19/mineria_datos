import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Cargar el dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Asegurémonos de que las columnas necesarias existan
print(df.columns)

# Seleccionamos las columnas necesarias y eliminamos los valores nulos
df = df[['popularity', 'vote_average']].dropna()

# Visualización de la relación entre Popularidad y Promedio de Votos (Diagrama de dispersión)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='popularity', y='vote_average')
plt.title('Relación entre Popularidad y Promedio de Votos')
plt.xlabel('Popularidad')
plt.ylabel('Promedio de Votos')
# Guardar la gráfica en la carpeta actual
plt.savefig('scatter_popularity_vs_vote_average.png')
plt.show()

# Crear el modelo de regresión lineal
X = df[['popularity']]  # variable independiente
y = df['vote_average']  # variable dependiente

model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Añadir la línea de regresión al gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='popularity', y='vote_average', color='blue', label='Datos Reales')
plt.plot(df['popularity'], y_pred, color='red', label='Línea de Regresión')
plt.title('Modelo de Regresión Lineal: Popularidad vs Promedio de Votos')
plt.xlabel('Popularidad')
plt.ylabel('Promedio de Votos')
plt.legend()
# Guardar la gráfica en la carpeta actual
plt.savefig('regression_line_popularity_vs_vote_average.png')
plt.show()

# Calcular el R² (coeficiente de determinación)
r2_score = model.score(X, y)
print(f"R² (Coeficiente de Determinación): {r2_score}")

# Calcular la correlación entre 'popularity' y 'vote_average'
correlation = df.corr()
print(f"Correlación entre Popularidad y Promedio de Votos: {correlation.loc['popularity', 'vote_average']}")

