import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PerceptronIncremental:
    def __init__(self, tasa_aprendizaje=0.1, epocas=10):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos = None
        self.historial_precision = []

    def ajustar(self, X, Y):
        num_muestras, num_caracteristicas = X.shape
        self.pesos = np.zeros(num_caracteristicas)
        Y_ = np.array([1 if i > 0 else 0 for i in Y])

        for _ in range(self.epocas):
            for idx, x_i in enumerate(X):
                condicion = np.dot(x_i, self.pesos)
                prediccion = 1 if condicion > 0 else 0
                actualizacion = self.tasa_aprendizaje * (Y_[idx] - prediccion)
                self.pesos += actualizacion * x_i

            # Registrar precisión después de cada época
            predicciones = self.predecir(X)
            precision_actual = puntuacion_precision(Y_, predicciones)
            self.historial_precision.append(precision_actual)

    def predecir(self, X):
        prediccion = np.dot(X, self.pesos)
        return [1 if i > 0 else 0 for i in prediccion]


def puntuacion_precision(y_verdadero, y_predicho):
    predicciones_correctas = sum(
        [verdadero == predicho for verdadero, predicho in zip(y_verdadero, y_predicho)]
    )
    return predicciones_correctas / len(y_verdadero)


def error_rate(y_verdadero, y_predicho):
    precision = puntuacion_precision(y_verdadero, y_predicho)
    return 1 - precision


def binarizar_datos(datos):
    for columna in datos.columns[:-1]:
        valor_mediano = datos[columna].median()
        datos[columna] = (datos[columna] > valor_mediano).astype(int)
    return datos


# Cargar y preprocesar los datos
datos = pd.read_csv("breast_cancer_data.csv")
datos = binarizar_datos(datos)

# Dividir los datos
proporcion_division = 0.8
indice_division = int(len(datos) * proporcion_division)
datos_entrenamiento = datos.iloc[:indice_division]
datos_prueba = datos.iloc[indice_division:]

# Separar características y etiquetas
X_entrenamiento = datos_entrenamiento.drop(columns="Label").values
Y_entrenamiento = datos_entrenamiento["Label"].values
X_prueba = datos_prueba.drop(columns="Label").values
Y_prueba = datos_prueba["Label"].values

# Entrenar y visualizar el progreso
perceptron_incremental = PerceptronIncremental(epocas=100)
perceptron_incremental.ajustar(X_entrenamiento, Y_entrenamiento)

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(perceptron_incremental.historial_precision) + 1),
    perceptron_incremental.historial_precision,
    marker="o",
)
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Progreso del entrenamiento del Perceptrón")
plt.grid(True)
plt.show()


# Realizar predicciones en el conjunto de prueba
predicciones = perceptron_incremental.predecir(X_prueba)

# Calcular la precisión y la tasa de error
precision = puntuacion_precision(Y_prueba, predicciones) * 100
tasa_error = error_rate(Y_prueba, predicciones) * 100

# Imprimir los resultados
print("Resultados del conjunto de prueba:")
for y_verdadero, y_predicho in zip(Y_prueba, predicciones):
    etiqueta_esperada = "Maligno" if y_verdadero == 1 else "Benigno"
    etiqueta_predicha = "Maligno" if y_predicho == 1 else "Benigno"
    print(f"Esperado: {etiqueta_esperada}, Predicho: {etiqueta_predicha}")

print(f"\nPrecisión del modelo: {precision:.2f}%")
print(f"Tasa de error del modelo: {tasa_error:.2f}%")
