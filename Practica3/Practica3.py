import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("tmdb_10000_movies.csv")

# Asegurar que las columnas clave estén limpias y en el tipo correcto
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date', 'popularity', 'vote_average', 'vote_count', 'original_language'])

# Configuración de estilo
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 1. Histograma de la popularidad
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True, color='skyblue')
plt.title("Distribución de la Popularidad")
plt.xlabel("Popularidad")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# 2. Boxplot de las calificaciones promedio
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['vote_average'], color='orange')
plt.title("Boxplot de Calificaciones Promedio")
plt.xlabel("Calificación Promedio")
plt.tight_layout()
plt.show()

# 3. Scatter plot entre popularidad y número de votos
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='popularity', data=df, alpha=0.5, edgecolor='w')
plt.title("Popularidad vs. Número de Votos")
plt.xlabel("Número de Votos")
plt.ylabel("Popularidad")
plt.tight_layout()
plt.show()

# 4. Pie chart de los 5 idiomas originales más comunes
plt.figure(figsize=(8, 8))
top_langs = df['original_language'].value_counts().nlargest(5)
plt.pie(top_langs, labels=top_langs.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Top 5 Idiomas Originales en Películas")
plt.tight_layout()
plt.show()

# 5. Gráfico de línea: cantidad de películas por año
df['release_year'] = df['release_date'].dt.year
movies_per_year = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x=movies_per_year.index, y=movies_per_year.values)
plt.title("Películas Lanzadas por Año")
plt.xlabel("Año")
plt.ylabel("Cantidad de Películas")
plt.tight_layout()
plt.show()

