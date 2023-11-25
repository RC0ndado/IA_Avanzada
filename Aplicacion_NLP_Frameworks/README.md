# Análisis de Sentimientos con Flask

A01379299 - Ricardo Ramírez Condado

Este proyecto es una aplicación web desarrollada con Flask que realiza un análisis de sentimientos en textos proporcionados por el usuario. Utiliza la biblioteca `transformers` de Hugging Face para procesar el texto y determinar su tono emocional.

## Requisitos Previos

Antes de ejecutar la aplicación, asegúrate de tener instalado Python 3.6 o superior. También necesitarás pip para la instalación de paquetes.

## Instalación

Para configurar y ejecutar la aplicación, sigue estos pasos:

1. **Clonar el Repositorio (Opcional):**
   Si tienes el código en un repositorio de Git, clónalo en tu máquina local. De lo contrario, asegúrate de tener todos los archivos del proyecto en tu computadora.

```
git clone https://github.com/RC0ndado/IA_Avanzada.git
cd Aplicacion_NLP_Frameworks
```

2. **Crear un Entorno Virtual:**
   Es recomendable utilizar un entorno virtual para manejar las dependencias.

```
python -m venv venv
```

Activa el entorno virtual:

- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

3. **Instalar Dependencias:**
   Instala todas las dependencias necesarias utilizando pip:

```
pip install flask flask-bootstrap transformers torch
```

4. **Configuración Adicional (Opcional):**
   Si tu aplicación necesita configuraciones adicionales (como variables de entorno), asegúrate de establecerlas en este punto.

## Ejecución

Para ejecutar la aplicación:

```
python app.py
```

Esto iniciará un servidor de desarrollo local. Abre un navegador y visita `http://127.0.0.1:5000` para interactuar con la aplicación.

## Uso

Una vez que la aplicación esté en funcionamiento, puedes utilizarla siguiendo estos pasos:

1. Escribe o pega un texto en el campo proporcionado en la página web.
2. Haz clic en el botón 'Submit' para enviar el texto.
3. La aplicación mostrará el análisis de sentimientos del texto proporcionado.

## Vídeo Demostración

En el siguiente vídeo se muestra la funcionalidad en aplicación del siguiente código:


https://github.com/RC0ndado/IA_Avanzada/assets/81990698/6cb3f6d2-4bdb-4de5-8a7a-318f4a3f8053
