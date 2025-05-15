import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
import os

# Descargar los stopwords si es la primera vez
nltk.download('stopwords')

# Leer el dataset limpio
df = pd.read_csv(r"C:\Users\joele\OneDrive\Documentos\FCFM\8semestre\mineria_datos\csv\tmdb_10000_movies_cleaned.csv")

# Asegurar que la columna 'overview' existe y no tiene nulos
text_data = df['overview'].dropna().astype(str)

# Preprocesamiento del texto:
stop_words = set(stopwords.words('english'))  # puedes cambiar a 'spanish' si fuera el caso
text_clean = ""

for overview in text_data:
    # Convertir a minúsculas
    overview = overview.lower()
    # Eliminar signos de puntuación
    overview = overview.translate(str.maketrans('', '', string.punctuation))
    # Eliminar stopwords
    words = overview.split()
    filtered_words = [word for word in words if word not in stop_words]
    text_clean += " ".join(filtered_words) + " "

# Generar la nube de palabras
wordcloud = WordCloud(width=1200, height=600, background_color='white', max_words=200).generate(text_clean)

# Mostrar la nube de palabras
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Overview de Películas', fontsize=16)

# Guardar imagen en la misma carpeta que el script
output_path = os.path.join(os.getcwd(), "wordcloud_overview.png")
plt.savefig(output_path)
plt.close()

print(f"Nube de palabras generada y guardada como {output_path}")
