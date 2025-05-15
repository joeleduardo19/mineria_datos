import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar dataset
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Convertir 'release_date' a datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Filtrar filas con fecha y popularity no nulos
df = df.dropna(subset=['release_date', 'popularity'])

# Ordenar por fecha
df = df.sort_values('release_date')

# Crear índice numérico para tiempo (puede ser días desde la fecha mínima)
df['time_index'] = (df['release_date'] - df['release_date'].min()).dt.days

# Variable predictora X (tiempo), variable objetivo y (popularity)
X = df['time_index'].values.reshape(-1,1)
y = df['popularity'].values

# División 80% entrenamiento, 20% prueba
split_index = int(0.8 * len(df))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción en test
y_pred = model.predict(X_test)

# Métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')

# Graficar serie temporal real vs predicción
plt.figure(figsize=(12,6))
plt.plot(df['release_date'][:split_index], y_train, label='Entrenamiento')
plt.plot(df['release_date'][split_index:], y_test, label='Real')
plt.plot(df['release_date'][split_index:], y_pred, label='Predicción', linestyle='--')
plt.title('Forecasting Popularity con Regresión Lineal')
plt.xlabel('Fecha de Estreno')
plt.ylabel('Popularity')
plt.legend()
plt.grid(True)
plt.savefig('forecasting_popularity.png')
plt.close()

# Predicción para los próximos 6 meses después del último dato
from datetime import timedelta

last_day = df['release_date'].max()
future_days = np.array([(last_day + timedelta(days=i)).toordinal() for i in range(1, 181)])  # 6 meses ~180 días

# Ajustar futuro a time_index
future_time_index = future_days - df['release_date'].min().toordinal()
future_time_index = future_time_index.reshape(-1,1)

future_pred = model.predict(future_time_index)

# Fechas futuras
future_dates = [last_day + timedelta(days=i) for i in range(1, 181)]

# Mostrar predicciones futuras
df_future = pd.DataFrame({'Date': future_dates, 'Predicted_Popularity': future_pred})
print("\nPredicciones futuras de Popularity (6 meses):")
print(df_future.head())

# Graficar toda la serie con predicción futura
plt.figure(figsize=(12,6))
plt.plot(df['release_date'], y, label='Datos reales')
plt.plot(df_future['Date'], df_future['Predicted_Popularity'], label='Predicción futura', linestyle='--', marker='o')
plt.title('Serie Temporal Popularity con Predicción Futura')
plt.xlabel('Fecha de Estreno')
plt.ylabel('Popularity')
plt.legend()
plt.grid(True)
plt.savefig('popularity_con_prediccion_futura.png')
plt.close()
