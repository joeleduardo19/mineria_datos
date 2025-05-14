import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Asegurarse de que los valores nulos no interfieran
df = df.dropna(subset=['popularity', 'vote_average', 'vote_count', 'original_language'])

# Pie Chart: Idiomas más comunes
top_languages = df['original_language'].value_counts().head(5)
plt.figure(figsize=(6, 6))
plt.pie(top_languages, labels=top_languages.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Idiomas Originales en las Películas')
plt.savefig("piechart_languages.png")
plt.show()

# Histogramas de variables numéricas 
numeric_cols = ['popularity', 'vote_average', 'vote_count']
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"hist_{col}.png")
    plt.show()

# Boxplots agrupados por idioma 
for col in numeric_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='original_language', y=col, data=df)
    plt.title(f'{col} por Idioma Original')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}_by_language.png")
    plt.show()

# Diagramas de líneas: promedio por idioma 
for col in numeric_cols:
    mean_per_lang = df.groupby('original_language')[col].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.plot(mean_per_lang.index, mean_per_lang.values, marker='o')
    plt.title(f'Promedio de {col} por Idioma')
    plt.xlabel('Idioma')
    plt.ylabel(f'{col} promedio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lineplot_mean_{col}_by_language.png")
    plt.show()

# Diagrama de dispersión 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vote_count', y='popularity', hue='original_language', alpha=0.6)
plt.title('Relación entre Número de Votos y Popularidad')
plt.xlabel('Número de Votos')
plt.ylabel('Popularidad')
plt.legend(title='Idioma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("scatter_votes_vs_popularity.png")
plt.show()
