# Importaciones de Flask y otras bibliotecas necesarias
from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from transformers import (
    pipeline,
    DistilBertTokenizer,
)  # Importa DistilBertTokenizer aquí
import time

# Inicialización de la aplicación Flask y configuración de Bootstrap
app = Flask(__name__)
Bootstrap(app)

# Configuración del pipeline de análisis de sentimientos usando Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Cargar el tokenizador para el modelo específico
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Ruta de la página de inicio de la aplicación web
@app.route("/")
def index():
    '''Renderiza la página de inicio con 
    valores predeterminados para las variables'''
    return render_template("index.html", sentiment_analysis=None, received_text=None)


# Ruta para el análisis de sentimientos, se activa con solicitudes POST
@app.route("/analyse", methods=["POST"])
def analyse():
    '''Verifica si el método de 
    la solicitud es POST'''
    if request.method == "POST":
        # Obtiene el texto del formulario
        rawtext = request.form["rawtext"]
        # Realiza el análisis de sentimientos en el texto
        analysis_results = perform_analysis(rawtext)
        # Renderiza la página de inicio con los resultados del análisis
        return render_template("index.html", **analysis_results)
    # Si no es una solicitud POST, renderiza la página de inicio con un mensaje de error
    return render_template(
        "index.html",
        error="Invalid Request",
        sentiment_analysis=None,
        received_text=None,
    )


def perform_analysis(text):
    '''Función para realizar 
    el análisis de sentimientos
    en el texto'''
    start = time.time()

    # Truncar el texto usando el tokenizador
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    truncated_text = tokenizer.decode(inputs["input_ids"][0])

    # Utiliza el pipeline para realizar el análisis de sentimientos
    sentiment_result = sentiment_pipeline(truncated_text)

    # Marca el tiempo de finalización del análisis
    end = time.time()
    # Devuelve los resultados del análisis
    return {
        "received_text": truncated_text,
        "sentiment_analysis": sentiment_result[0] if sentiment_result else None,
        "processing_time": end - start,
    }

# Punto de entrada principal para ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)
