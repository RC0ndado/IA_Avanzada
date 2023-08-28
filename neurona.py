# ----------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje
# máquina sin el uso de un framework. (Portafolio Implementación)
#
# Date: 27-Aug-2022
# Authors:
#           A01379299 Ricardo Ramírez Condado
#
# Fecha de creación: 24/08/2022
# Última actualización: 27/08/2022
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


if __name__ == "__main__":
    # Cargar y preprocesar los datos
    datos = pd.read_csv("breast_cancer_data.csv")
    datos = binarizar_datos(datos)

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    proporcion_division = 0.8
    indice_division = int(len(datos) * proporcion_division)
    datos_entrenamiento = datos.iloc[:indice_division]
    datos_prueba = datos.iloc[indice_division:]

    # Separar características y etiquetas
    X_entrenamiento = datos_entrenamiento.drop(columns="Label").values.tolist()
    Y_entrenamiento = datos_entrenamiento["Label"].values.tolist()
    X_prueba = datos_prueba.drop(columns="Label").values.tolist()
    Y_prueba = datos_prueba["Label"].values.tolist()

    # Entrenar la Neurona MP y obtener el umbral óptimo
    umbral_optimo = ajustar(X_entrenamiento, Y_entrenamiento)

    # Predecir en el conjunto de prueba
    Y_predicho = predecir(X_prueba, umbral_optimo)

    # Imprimir predicciones individuales para cada punto de datos
    for y_verdadero, y_predicho in zip(Y_prueba, Y_predicho):
        etiqueta_esperada = "maligno" if y_verdadero == 1 else "benigno"
        etiqueta_predicha = "maligno" if y_predicho else "benigno"
        print(f"Esperado: {y_verdadero}, Predicho: {etiqueta_predicha}")

    # Calcular la precisión
    precision = puntuacion_precision(Y_prueba, Y_predicho) * 100
    print(f"\nPrecisión del modelo: {precision:.2f}%\n")
