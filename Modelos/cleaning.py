import pandas as pd
import numpy as np
from collections import Counter
import re                         
import spacy
import string 
from tensorflow import keras

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from nltk.corpus import stopwords
nltk.download('stopwords')
", ".join(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize
nltk.download('punkt')
## funcion para leer data y limpiar

def read_excel_data(file) -> pd.DataFrame:
    try:
        # Leer el archivo Excel directamente desde el objeto de archivo subido
        raw_data = pd.read_excel(file)

        # Verificar si el archivo tiene datos
        if not raw_data.empty and 'text' in raw_data.columns:
            return raw_data
        else:
            print('El archivo está vacío o falta la columna esperada.')
            return None

    except Exception as e:
        print(f'Error al procesar el archivo Excel: {e}')
        return None

def transformar_nuevo(df, model_mini, columns=['text']):
    # Procesamiento de cada columna de texto
    for column in columns:
        df[column] = df[column].astype(str).str.lower()
        df[column] = df[column].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
        df[column] = df[column].apply(lambda text: ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))]))
        df[column] = df[column].apply(lambda text: word_tokenize(text))
        df[column] = df[column].apply(lambda tokens: ' '.join([WordNetLemmatizer().lemmatize(word) for word in tokens]))


    # Vectorización
    resumen_fraseo_nuevo = []
    for frase in df['text']:
        tokens = word_tokenize(frase)
        tokens_en_vocabulario = [token for token in tokens if token in model_mini.wv.index_to_key]

        if tokens_en_vocabulario:
            vectores_en_vocabulario = [model_mini.wv[token] for token in tokens_en_vocabulario]
            promedios = np.mean(vectores_en_vocabulario, axis=0)
            resumen_fraseo_nuevo.append(promedios)
        else:
            resumen_fraseo_nuevo.append(np.zeros(model_mini.vector_size))

    X_nuevo = pd.DataFrame(resumen_fraseo_nuevo)
    return X_nuevo 

def top_palabras(df, num_palabras=20):
    # Unir todos los textos en una sola cadena
    texto_unido = ' '.join(df['text'])

    # Limpiar y tokenizar el texto
    palabras = re.findall(r'\w+', texto_unido.lower())

    # Filtrar las stopwords
    palabras = [palabra for palabra in palabras if palabra not in stop_words]

    # Contar y obtener las palabras más frecuentes
    frecuencia_palabras = Counter(palabras)
    top_palabras = frecuencia_palabras.most_common(num_palabras)

    return top_palabras