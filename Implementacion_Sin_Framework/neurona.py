# last version----------------------------------------------------------
# Módulo 2 Implementación de una técnica de aprendizaje
# máquina sin el uso de un framework. (Portafolio Implementación)
#
# Date: 27-Aug-2022
# Authors:
#           A01379299 Ricardo Ramírez Condado
#
# Fecha de creación: 24/08/2022
# Última actualización: 30/08/2023
# ----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def matriz_confusion(Y_verdadero, Y_predicho):
    """
    Genera una matriz de confusión a partir de las etiquetas verdaderas y las predichas.

    Y_verdadero: Lista de etiquetas verdaderas
    Y_predicho: Lista de etiquetas predichas por el modelo

    Retorna: Matriz de confusión en formato 2x2
    """

    # Cálculo de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos
    tp = sum([y_v and y_p for y_v, y_p in zip(Y_verdadero, Y_predicho)])
    tn = sum([not y_v and not y_p for y_v, y_p in zip(Y_verdadero, Y_predicho)])
    fp = sum([not y_v and y_p for y_v, y_p in zip(Y_verdadero, Y_predicho)])
    fn = sum([y_v and not y_p for y_v, y_p in zip(Y_verdadero, Y_predicho)])

    # Retornar la matriz de confusión
    return np.array([[tp, fp], [fn, tn]])


def modelo_neurona_mp(x, umbral):
    """
    Define el modelo de la Neurona MP.

    x: Vector de entrada
    umbral: Valor de umbral para activar la neurona

    Retorna: Verdadero si la suma de x es mayor o igual al umbral, Falso en caso contrario
    """
    return sum(x) >= umbral


def predecir(X, umbral):
    """
    Predice la salida para un conjunto dado de vectores de entrada usando el modelo Neurona MP.

    X: Matriz de datos de entrada
    umbral: Valor de umbral para activar la neurona

    Retorna: Lista de predicciones (Verdadero o Falso)
    """
    return [modelo_neurona_mp(x, umbral) for x in X]


def ajustar(X, Y):
    """
    Ajusta el modelo de la Neurona MP a los datos de entrenamiento encontrando el umbral óptimo.

    X: Matriz de datos de entrada
    Y: Lista de etiquetas verdaderas

    Retorna: Umbral óptimo para el modelo Neurona MP
    """

    precisiones = []

    # Itera sobre todos los posibles umbrales (desde 0 hasta la longitud de un vector de entrada)
    for umbral in range(len(X[0]) + 1):
        Y_pred = predecir(X, umbral)
        precisiones.append(puntuacion_precision(Y_pred, Y))

    # Retorna el umbral que obtuvo la máxima precisión
    return precisiones.index(max(precisiones))


def puntuacion_precision(y_verdadero, y_predicho):
    """
    Calcula la precisión de las predicciones.

    y_verdadero: Lista de etiquetas verdaderas
    y_predicho: Lista de etiquetas predichas por el modelo

    Retorna: Precisión de las predicciones
    """
    # Cálculo de las predicciones correctas
    predicciones_correctas = sum(
        [verdadero == predicho for verdadero, predicho in zip(y_verdadero, y_predicho)]
    )
    return predicciones_correctas / len(y_verdadero)


def binarizar_datos(datos):
    """
    Binariza la columna "Label" del DataFrame y también binariza las características usando la mediana.

    datos: DataFrame original

    Retorna: DataFrame binarizado
    """
    for columna in datos.columns[:-1]:  # Excluyendo la columna 'Label'
        valor_mediano = datos[columna].median()
        datos[columna] = (datos[columna] > valor_mediano).astype(int)
    return datos


if __name__ == "__main__":
    # Carga y preprocesamiento de los datos
    datos = pd.read_csv("breast_cancer_data.csv")
    datos = binarizar_datos(datos)

    # Separación de características y etiquetas
    X = datos.drop(columns="Label").values
    Y = datos["Label"].values

    # Inicialización de la validación cruzada
    kfold = KFold(n_splits=5, shuffle=True)

    # Listas para almacenar métricas
    precisiones = []
    recalls = []
    f1_scores = []

    print("Resultados de validación cruzada:")
    print("---------------------------------")

    # Listas para almacenar todas las predicciones y métricas
    todas_predicciones_texto = []
    todas_metricas = []

    # Iterar sobre cada fold en la validación cruzada
    for train_index, test_index in kfold.split(X):
        X_entrenamiento, X_prueba = X[train_index], X[test_index]
        Y_entrenamiento, Y_prueba = Y[train_index], Y[test_index]

        # Ajustar el modelo y obtener el umbral óptimo
        umbral_optimo = ajustar(X_entrenamiento, Y_entrenamiento)

        # Realizar predicciones con el umbral óptimo
        Y_predicho = predecir(X_prueba, umbral_optimo)

        # Almacenar predicciones individuales
        predicciones_texto = []

        # Mostrar predicciones individuales
        for y_verdadero, y_predicho in zip(Y_prueba, Y_predicho):
            etiqueta_esperada = "maligno" if y_verdadero == 1 else "benigno"
            etiqueta_predicha = "maligno" if y_predicho else "benigno"
            predicciones_texto.append(
                f"Esperado: {etiqueta_esperada}, Predicho: {etiqueta_predicha}"
            )

        # Cálculo de métricas y almacenamiento de las mismas
        precision = precision_score(Y_prueba, Y_predicho)
        recall = recall_score(Y_prueba, Y_predicho)
        f1 = f1_score(Y_prueba, Y_predicho)

        # Mostrar métricas de esta iteración
        precisiones.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        todas_metricas.append(
            f"Precisión: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%"
        )

        print(
            f"Precisión: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%"
        )

        # Imprimir predicciones individuales
        for prediccion in predicciones_texto:
            print(prediccion)

        # Separador para la siguiente iteración
        print("\n" + "-" * 30)

        # Imprimir todas las métricas
        for metrica in todas_metricas:
            print(metrica)

        # Imprimir todas las métricas
        for metrica in todas_metricas:
            print(metrica)

        # Separador para la siguiente
        print("\n" + "-" * 30)

        # Imprimir matriz de confusión
        print("Matriz de confusión:")
        print(matriz_confusion(Y_prueba, Y_predicho))

    print("\nResumen:")
    print(f"Precisión media: {np.mean(precisiones)*100:.2f}%")
    print(f"Recall medio: {np.mean(recalls)*100:.2f}%")
    print(f"F1 Score medio: {np.mean(f1_scores)*100:.2f}%")
