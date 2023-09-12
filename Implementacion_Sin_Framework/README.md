# Perceptrón de McCulloch-Pitts

## Descripción:

El Perceptrón de McCulloch-Pitts es uno de los modelos más simples de neuronas artificiales. Trabaja con entradas binarias y determina la activación basándose en un umbral. El algoritmo se ha implementado para predecir si un tumor es benigno o maligno basándose en características relacionadas con diagnósticos de cáncer de mama.

## Mecanismo:

- Se establece un umbral basado en los datos de entrenamiento.
- Para una entrada dada, si la suma de las características (ponderada por los pesos) supera el umbral, la neurona se activa; de lo contrario, no se activa.
- En este modelo, la activación significa que el tumor es maligno, mientras que la no activación significa que es benigno.

## Objetivo de código:

Se espera que el código sea capaz de detectar si un paciente tiene cancer de mama en base a una cantidad de datos provenientes de un excel.

**_Nota: Los datos que se generaron al azar, sin embargo, considerando que se intenta modelar datos de la vida real, se generaron en inspirandose y condicionandolos a que se parecieran un poco a los datos provenientes de sklearn.datasets.load_breast_cancer.html._**

## Cómo correr el código:

1. Clona el repositorio o descarga el archivo .zip.
   ![image](https://github.com/RC0ndado/IA_Avanzada/blob/main/assets/ss1.png)

2. Abre la carpeta en tu entorno local (asegúrate de tener Python instalado, junto con las bibliotecas `pandas` y `numpy`).
   ![image](https://github.com/RC0ndado/IA_Avanzada/blob/main/assets/ss2.png)

- Nota: Si no tienes la biblioteca instalada, usa `pip install pandas`, o si no tienes la biblioteca instalada de numpy, usa `pip install numpy`.

3. Ejecuta el código con el corredor de código o terminal. Las dos formas de ejecutarlo son:

   1. Escribe en tu terminal `python neurona.py `  
      ![image](https://github.com/RC0ndado/IA_Avanzada/blob/main/assets/ss3.png).

   2. O bien puedes usar el corredor de código:
      ![image](https://github.com/RC0ndado/IA_Avanzada/blob/main/assets/ss5.png).

4. Una vez que el programa se ejecute, te mostrará la precisión del modelo con respecto al conjunto de pruebas.
   ![image](https://github.com/RC0ndado/IA_Avanzada/blob/main/assets/ss4.png)
