# Importar las librerías necesarias
import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies.csv")


# Ver las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Verificar si hay valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Eliminar duplicados
df = df.drop_duplicates()

# Convertir la columna 'release_date' a tipo fecha
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Eliminar filas con fechas nulas
df = df.dropna(subset=['release_date'])

# Verificar el estado después de la limpieza
print("\nEstado después de la limpieza:")
print(df.info())

# Guardar el archivo limpio
df.to_csv('tmdb_10000_movies_cleaned.csv', index=False)
