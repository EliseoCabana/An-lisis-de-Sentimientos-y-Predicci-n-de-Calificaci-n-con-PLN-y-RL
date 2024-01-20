from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from cleaning import  transformar_nuevo, top_palabras

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files or file.filename == '':
            return 'No file part or no file selected'

        try:
            # Leer directamente el archivo Excel
            df = pd.read_excel(file)

            # Verificar si el archivo tiene datos y contiene la columna 'text'
            if df.empty or 'text' not in df.columns:
                return 'El archivo está vacío o falta la columna esperada'

        except Exception as e:
            return f'Error al procesar el archivo Excel: {e}'
        
        # Cargar los modelos
        model_vect = joblib.load('Modelo_VECT.joblib')
        modelo_pln = joblib.load('Modelo1_PLN.joblib')

        df_limpio = transformar_nuevo(df, model_vect)  # Utiliza el modelo de vectorización

        # Realizar predicciones
        predicciones = modelo_pln.predict(df_limpio)
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=predicciones)

        # Personalizar colores
        colores = ['green' if x == 'Positive' else 'red' for x in predicciones]
        sns.countplot(x=predicciones, palette=colores)

        # Añadir etiquetas de datos
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='gray', xytext=(0, 5),
                        textcoords='offset points')

        plt.title('Distribución de Etiquetas Predichas')

        # Guardar el gráfico en un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

        html = f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Resultado de Análisis</title>
        </head>
        <body>
            <img src="data:image/png;base64,{plot_url}" />
        </body>
        </html>
        '''
        return render_template_string(html)

    # HTML para cargar archivo, incluyendo todo el diseño y estilos que proporcionaste
    return '''
        <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Analiza los comentarios de tu App y Predice su calificación</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                padding: 50px;
            }

            .title-container {
                background-color: #4CAF50; /* Un verde no muy fuerte */
                color: white;
                margin: 0 auto;
                width: fit-content;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            form {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                display: inline-block;
            }

            input[type=file] {
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="title-container">
            <h1>Analiza los comentarios de tu App y Predice su calificación :)</h1>
        </div>
        <p class="instruction">Sube tu archivo Excel solo de una columna que se llame 'text'</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Subir Archivo">
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)