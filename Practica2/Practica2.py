import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Verificar columnas
print("Columnas del dataset:")
print(df.columns)

# Estadísticas descriptivas básicas
print("\nEstadísticas descriptivas:")
print(df[['vote_average', 'vote_count', 'popularity']].describe())

# Medidas de tendencia central y dispersión
for column in ['vote_average', 'vote_count', 'popularity']:
    print(f"\n--- {column.upper()} ---")
    print(f"Media: {df[column].mean():.2f}")
    print(f"Mediana: {df[column].median():.2f}")
    print(f"Moda: {df[column].mode().values}")
    print(f"Varianza: {df[column].var():.2f}")
    print(f"Desviación estándar: {df[column].std():.2f}")

# Agrupar por idioma original y obtener estadísticas
grouped_lang = df.groupby('original_language')[['vote_average', 'vote_count', 'popularity']].mean().sort_values(by='popularity', ascending=False)
print("\nEstadísticas agrupadas por idioma original (promedios):")
print(grouped_lang)

# Histograma de popularidad
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Distribución de Popularidad')
plt.xlabel('Popularidad')
plt.ylabel('Frecuencia')
plt.savefig("hist_popularity.png")
plt.show()

# Boxplot de puntuación por idioma
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='original_language', y='vote_average')
plt.title('Distribución de la Calificación Promedio por Idioma')
plt.xlabel('Idioma Original')
plt.ylabel('Calificación Promedio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_vote_avg_by_lang.png")
plt.show()

# Diagrama de dispersión: Popularidad vs. Número de votos
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vote_count', y='popularity', hue='original_language', alpha=0.6)
plt.title('Relación entre Número de Votos y Popularidad')
plt.xlabel('Número de Votos')
plt.ylabel('Popularidad')
plt.legend(title='Idioma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("scatter_vote_count_vs_popularity.png")
plt.show()
