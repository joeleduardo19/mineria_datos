import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Filtrar columnas relevantes y eliminar nulos
df = df[['original_language', 'popularity', 'vote_average', 'vote_count']].dropna()

# Seleccionar solo los 3 idiomas más frecuentes para pruebas estadísticas
top_langs = df['original_language'].value_counts().head(3).index.tolist()
df_filtered = df[df['original_language'].isin(top_langs)]

# Mostrar los idiomas seleccionados
print("Idiomas seleccionados para análisis:", top_langs)

# Visualización previa: Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='original_language', y='popularity', data=df_filtered)
plt.title('Distribución de Popularidad por Idioma')
plt.savefig("boxplot_popularity_test.png")
plt.show()

# -------------------------------------------
# 1. ANOVA (Análisis de Varianza) - Popularidad por idioma
# -------------------------------------------
grupo1 = df_filtered[df_filtered['original_language'] == top_langs[0]]['popularity']
grupo2 = df_filtered[df_filtered['original_language'] == top_langs[1]]['popularity']
grupo3 = df_filtered[df_filtered['original_language'] == top_langs[2]]['popularity']

anova_result = stats.f_oneway(grupo1, grupo2, grupo3)
print("ANOVA:")
print(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")

# -------------------------------------------
# 2. Prueba de T - Popularidad entre dos idiomas
# -------------------------------------------
t_result = stats.ttest_ind(grupo1, grupo2, equal_var=False)
print("\nPrueba T (entre primeros dos idiomas):")
print(f"t-statistic: {t_result.statistic:.4f}, p-value: {t_result.pvalue:.4f}")

# -------------------------------------------
# 3. Kruskal-Wallis (si no hay normalidad)
# -------------------------------------------
kruskal_result = stats.kruskal(grupo1, grupo2, grupo3)
print("\nKruskal-Wallis:")
print(f"H-statistic: {kruskal_result.statistic:.4f}, p-value: {kruskal_result.pvalue:.4f}")

# -------------------------------------------
# Interpretación simple de resultados
# -------------------------------------------
print("\nInterpretación:")

if anova_result.pvalue < 0.05:
    print("- La prueba ANOVA sugiere diferencias significativas en la popularidad media entre los idiomas.")
else:
    print("- ANOVA no detectó diferencias significativas.")

if t_result.pvalue < 0.05:
    print("- La prueba T sugiere diferencia significativa entre los dos primeros idiomas.")
else:
    print("- La prueba T no sugiere diferencia significativa.")

if kruskal_result.pvalue < 0.05:
    print("- La prueba de Kruskal-Wallis también confirma diferencias significativas.")
else:
    print("- Kruskal-Wallis no muestra diferencias significativas.")
