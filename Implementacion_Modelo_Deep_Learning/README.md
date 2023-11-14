# Fashion Product Classifier

## Descripción

Este proyecto es un clasificador de productos de moda que utiliza un modelo de deep learning para clasificar imágenes de moda en categorías específicas. El modelo ha sido entrenado utilizando un conjunto de datos de Kaggle que incluye una gran variedad de productos de moda.

## Instalación

Para ejecutar este proyecto, necesitarás Python instalado en tu sistema, así como las siguientes librerías de Python:

- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib (opcional, para visualización de datos)

Puedes instalar todas las dependencias necesarias con:

```
pip install tensorflow keras pandas numpy matplotlib
```

## Descarga de Datos

El conjunto de datos puede ser descargado desde [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (si el enlace no llegara a funcionar, el dataset se puede encontrar con el nombre de "Fashion Product Images (Small)"). Necesitarás una cuenta de Kaggle y la API de Kaggle configurada en tu sistema para poder descargar los datos directamente.

Una vez configurada la API de Kaggle, puedes ejecutar el siguiente comando para descargar el dataset:

```
kaggle datasets download -d paramaggarwal/fashion-product-images-small
```

Descomprime el archivo descargado en el directorio del proyecto para continuar.

## Uso

Para entrenar y evaluar el modelo, simplemente ejecuta el notebook `Fashion_Classifier.ipynb` en un entorno Jupyter. El notebook contiene todas las instrucciones paso a paso y celdas de código que te guiarán a través del proceso de entrenamiento y evaluación del modelo.

### Pasos para Ejecutar el Notebook:

1. Clonar el repositorio o descargar todos los archivos en tu máquina local.
2. Abrir `Fashion_Classifier.ipynb` en Jupyter Notebook o JupyterLab.
3. Ejecutar las celdas en orden, siguiendo las instrucciones y comentarios proporcionados.

## Estructura del Proyecto

El proyecto incluye los siguientes archivos y carpetas principales:

- `Fashion_Classifier.ipynb`: Notebook de Jupyter con todo el código y la documentación.
- `model.h5`: El modelo de Keras entrenado guardado en disco.
- `images/`: Directorio donde se deben colocar las imágenes del conjunto de datos una vez descargadas.
- `styles/`: Directorio que contiene los metadatos de las imágenes.

## Contribuir

Las contribuciones son bienvenidas, y cualquier feedback o sugerencias serán muy apreciados. Si deseas contribuir al proyecto, por favor haz un 'fork' del repositorio y utiliza una 'pull request' para proponer tus cambios.

## Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).
