import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('tmdb_10000_movies_cleaned.csv')

# Asegurémonos de que las columnas que estamos utilizando existan en el dataset
print(df.columns)

# Calcular estadísticas descriptivas para las columnas numéricas
print(df[['budget', 'revenue', 'vote_average', 'popularity']].describe())

# Calcular la media, mediana, moda, varianza y desviación estándar para 'budget'
mean_budget = df['budget'].mean()
median_budget = df['budget'].median()
mode_budget = df['budget'].mode()[0]
variance_budget = df['budget'].var()
std_dev_budget = df['budget'].std()

print(f"Budget - Media: {mean_budget}, Mediana: {median_budget}, Moda: {mode_budget}, Varianza: {variance_budget}, Desviación Estándar: {std_dev_budget}")

# Agrupar por 'genre' y obtener estadísticas descriptivas
grouped_by_genre = df.groupby('genre')[['budget', 'revenue', 'vote_average', 'popularity']].describe()
print(grouped_by_genre)

# Graficar la distribución del presupuesto de las películas (histograma)
plt.figure(figsize=(10, 6))
sns.histplot(df['budget'], kde=True)
plt.title('Distribución del Presupuesto de las Películas')
plt.xlabel('Presupuesto')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la relación entre presupuesto y popularidad (diagrama de dispersión)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='budget', y='popularity')
plt.title('Relación entre Presupuesto y Popularidad')
plt.xlabel('Presupuesto')
plt.ylabel('Popularidad')
plt.show()
