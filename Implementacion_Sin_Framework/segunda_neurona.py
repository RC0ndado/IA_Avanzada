# ----------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje
# máquina sin el uso de un framework. (Portafolio Implementación)
#
# Date: 27-Aug-2022
# Authors:
#           A01379299 Ricardo Ramírez Condado
#
# Fecha de creación: 24/08/2022
# Última actualización: 28/08/2022
# ----------------------------------------------------------

import numpy as np
import csv
import pandas as pd


def modelo_neurona_mp(x, umbral):
    """
    Define el modelo de la Neurona MP.

    Parámetros:
    - x (lista): Un vector de entrada binario.
    - umbral (int): El umbral de activación para la neurona.

    Devuelve:
    - bool: True si la suma del vector de entrada es mayor o igual al umbral, False en caso contrario.
    """
    return sum(x) >= umbral


def predecir(X, umbral):
    """
    Predice la salida para un conjunto dado de vectores de entrada.

    Parámetros:
    - X (lista de listas): Una lista de vectores de entrada binarios.
    - umbral (int): El umbral de activación para la neurona.

    Devuelve:
    - lista: Una lista que contiene la salida binaria (True o False) para cada vector de entrada.
    """
    return [modelo_neurona_mp(x, umbral) for x in X]


def ajustar(X, Y):
    """
    Ajusta el modelo de la Neurona MP a los datos de entrenamiento encontrando el umbral óptimo.

    Parámetros:
    - X (lista de listas): Datos de entrada de entrenamiento.
    - Y (lista): Etiquetas verdaderas para los datos de entrenamiento.

    Devuelve:
    - int: Umbral óptimo para la neurona.
    """
    precisiones = []
    for umbral in range(len(X[0]) + 1):
        Y_pred = predecir(X, umbral)
        precisiones.append(puntuacion_precision(Y_pred, Y))
    return precisiones.index(max(precisiones))


def puntuacion_precision(y_verdadero, y_predicho):
    """
    Calcula la precisión de las predicciones.

    Parámetros:
    - y_verdadero (lista): Etiquetas verdaderas.
    - y_predicho (lista): Etiquetas predichas.

    Devuelve:
    - float: Precisión de las predicciones.
    """
    predicciones_correctas = sum(
        [verdadero == predicho for verdadero, predicho in zip(y_verdadero, y_predicho)]
    )
    return predicciones_correctas / len(y_verdadero)


def error_rate(y_verdadero, y_predicho):
    """Calcula el error rate de las predicciones."""
    precision = puntuacion_precision(y_verdadero, y_predicho)
    return 1 - precision


def binarizar_datos(datos):
    """
    Binariza las características del conjunto de datos en función de sus valores medianos.

    Parámetros:
    - datos (DataFrame): El conjunto de datos a binarizar.

    Devuelve:
    - DataFrame: El conjunto de datos binarizado.
    """
    for columna in datos.columns[:-1]:  # Excluyendo la columna 'Label'
        valor_mediano = datos[columna].median()
        datos[columna] = (datos[columna] > valor_mediano).astype(int)
    return datos


class PerceptronIncremental:
    def __init__(self, tasa_aprendizaje_inicial=0.1, epocas=100):
        self.tasa_aprendizaje_inicial = tasa_aprendizaje_inicial
        self.epocas = epocas
        self.pesos = None
        self.historial_precision = []

    def ajustar(self, X, Y):
        num_muestras, num_caracteristicas = X.shape
        self.pesos = np.random.randn(num_caracteristicas) * 0.01
        Y_ = np.array([1 if i > 0 else 0 for i in Y])

        for epoca in range(self.epocas):
            tasa_aprendizaje = self.tasa_aprendizaje_inicial / (
                1 + epoca
            )  # Tasa de aprendizaje adaptativa
            for idx, x_i in enumerate(X):
                condicion = np.dot(x_i, self.pesos)
                prediccion = 1 if condicion > 0 else 0
                actualizacion = tasa_aprendizaje * (Y_[idx] - prediccion)
                self.pesos += actualizacion * x_i

            predicciones = self.predecir(X)
            precision_actual = puntuacion_precision(Y_, predicciones)
            self.historial_precision.append(precision_actual)

    def predecir(self, X):
        prediccion = np.dot(X, self.pesos)
        return [1 if i > 0 else 0 for i in prediccion]


if __name__ == "__main__":
    datos = pd.read_csv("breast_cancer_data.csv")
    # No binarizamos los datos

    proporcion_division = 0.8
    indice_division = int(len(datos) * proporcion_division)
    datos_entrenamiento = datos.iloc[:indice_division]
    datos_prueba = datos.iloc[indice_division:]

    X_entrenamiento = datos_entrenamiento.drop(columns="Label").values
    Y_entrenamiento = datos_entrenamiento["Label"].values
    X_prueba = datos_prueba.drop(columns="Label").values
    Y_prueba = datos_prueba["Label"].values

    perceptron_incremental = PerceptronIncremental(
        tasa_aprendizaje_inicial=0.1, epocas=500
    )
    perceptron_incremental.ajustar(X_entrenamiento, Y_entrenamiento)

    # Realizar predicciones en el conjunto de prueba
    predicciones = perceptron_incremental.predecir(X_prueba)
    precision = puntuacion_precision(Y_prueba, predicciones) * 100
    tasa_error = error_rate(Y_prueba, predicciones) * 100

    print(f"\nPrecisión del modelo: {precision:.2f}%\n")
    print(f"Tasa de error del modelo: {tasa_error:.2f}%\n")
