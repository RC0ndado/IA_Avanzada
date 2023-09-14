# ----------------------------------------------------------
# Módulo 2 Uso de framework o biblioteca de aprendizaje
# máquina para la implementación de una solución (Portafolio Implementación)
#
# Date: 08-Aug-2022
# Authors:
#           A01379299 Ricardo Ramírez Condado
#
# Fecha de creación: 08/09/2022
# Última actualización: 11/09/2023
# ----------------------------------------------------------

"""
Diagnóstico de cáncer de mama con SVM con kernel lineal utilizando scikit-learn.

Este programa implementa el algoritmo SVM con un kernel lineal para predecir
si un tumor es benigno o maligno, basado en características del tumor.
El dataset es el conjunto de datos de cáncer de mama de Wisconsin.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
"""

# Importando las bibliotecas necesarias
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """
    Carga y devuelve el conjunto de datos de cáncer de mama.

    Returns:
        tuple: Una tupla que contiene las características (X) y
        las etiquetas/respuestas (y) del dataset.
    """
    return load_breast_cancer(return_X_y=True)


def split_data(X, y):
    """
    Divide el conjunto de datos en entrenamiento, prueba y validación.

    Args:
        X (numpy.ndarray): Conjunto de características del dataset.
        y (numpy.ndarray): Conjunto de etiquetas del dataset.

    Returns:
        tuple: Una tupla que contiene los conjuntos de datos de
        entrenamiento, prueba y validación.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Normaliza los conjuntos de datos utilizando StandardScaler.

    Args:
        X_train (numpy.ndarray): Conjunto de entrenamiento.
        X_val (numpy.ndarray): Conjunto de validación.
        X_test (numpy.ndarray): Conjunto de prueba.

    Returns:
        tuple: Conjuntos de datos normalizados.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def train_model(X_train, y_train):
    """
    Entrena y devuelve el modelo SVM.

    Args:
        X_train (numpy.ndarray): Conjunto de entrenamiento.
        y_train (numpy.ndarray): Etiquetas del conjunto de entrenamiento.

    Returns:
        SVC: Modelo SVM entrenado.
    """
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    return clf


def plot_custom_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """
    Esta función imprime y traza la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Loop over data dimensions and create text annotations.
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.show()


def evaluate_model(model, X, y):
    """
    Evalúa el modelo y muestra los resultados.

    Args:
        model (SVC): Modelo SVM entrenado.
        X (numpy.ndarray): Conjunto de datos a evaluar.
        y (numpy.ndarray): Etiquetas reales del conjunto de datos.

    Prints:
        Resultados de la evaluación, incluyendo reporte de clasificación,
        matriz de confusión y precisión.
    """
    y_pred = model.predict(X)
    print("Reporte de clasificación:")
    print(classification_report(y, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y, y_pred))
    print(f"Precisión: {accuracy_score(y, y_pred) * 100:.2f}%")

    # Gráfica de la Curva ROC
    y_prob = model.decision_function(X)  # obtener scores en lugar de predicciones
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=1, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()


def optimize_hyperparameters(X_train, y_train):
    """
    Optimiza los hiperparámetros del modelo SVM usando GridSearchCV.

    Args:
        X_train (numpy.ndarray): Conjunto de entrenamiento.
        y_train (numpy.ndarray): Etiquetas del conjunto de entrenamiento.

    Returns:
        GridSearchCV: Modelo SVM con hiperparámetros optimizados.
    """
    # Definir los parámetros a ajustar
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["linear", "rbf"],
    }

    # Usar GridSearchCV
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    # Mostrar los mejores parámetros
    print("Mejores parámetros encontrados: ", grid.best_params_)
    return grid


def plot_learning_curve(estimator, X, y, title="Learning Curves"):
    """
    Plota las curvas de aprendizaje para un estimador.

    Args:
        estimator (estimator object): El objeto modelo (en este caso, SVM).
        X (numpy.ndarray): Conjunto de características.
        y (numpy.ndarray): Conjunto de etiquetas.
        title (str, optional): Título para el gráfico. Por defecto "Learning Curves".
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    plt.show()


def main():
    """
    Función principal para ejecutar el flujo del programa.

    - Carga el dataset.
    - Divide el dataset.
    - Normaliza el dataset.
    - Entrena el modelo SVM.
    - Evalúa el modelo en los conjuntos de validación y prueba.
    """
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    model = train_model(X_train, y_train)

    # Defining parameter grid for C
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    # Using GridSearchCV to find the best parameter
    grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Getting the best parameter and best score
    best_C = grid_search.best_params_["C"]
    best_score = grid_search.best_score_

    best_C, best_score

    # Generating learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
    )

    # Calculating mean and standard deviation for train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plotting the learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.plot(train_sizes, val_mean, label="Validation score", color="green")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        color="blue",
        alpha=0.2,
    )
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, color="green", alpha=0.2
    )
    plt.title("Learning Curves for the SVM model")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # Preparing data for visualization
    data_lengths = [len(X_train), len(X_val), len(X_test)]
    labels = ["Training", "Validation", "Test"]

    # Plotting the distribution of data
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data_lengths, color=["blue", "green", "red"])
    plt.ylabel("Number of samples")
    plt.title("Distribution of Data between Training, Validation, and Test Sets")
    plt.show()

    # Reducing the CV and using a more spaced grid
    param_grid_reduced = {"C": [0.01, 0.1, 1, 10, 100]}

    # Using GridSearchCV to find the best parameter with reduced grid and CV
    grid_search_reduced = GridSearchCV(
        SVC(kernel="linear"), param_grid_reduced, cv=3, n_jobs=-1
    )
    grid_search_reduced.fit(X_train, y_train)

    # Getting the best parameter and best score from the reduced grid search
    best_C_reduced = grid_search_reduced.best_params_["C"]
    best_score_reduced = grid_search_reduced.best_score_

    best_C_reduced, best_score_reduced

    # Training the optimized model
    optimized_model = SVC(kernel="linear", C=best_C_reduced)
    optimized_model.fit(X_train, y_train)

    # Predictions using the original and optimized model
    y_pred_original = model.predict(X_val)
    y_pred_optimized = optimized_model.predict(X_val)

    # Plotting confusion matrices
    plt.figure(figsize=(14, 6))

    # Original Model
    plt.subplot(1, 2, 1)
    plot_custom_confusion_matrix(
        y_val, y_pred_original, classes=["Benign", "Malignant"]
    )
    plt.title("Original Model")

    # Optimized Model
    plt.subplot(1, 2, 2)
    plot_custom_confusion_matrix(
        y_val, y_pred_optimized, classes=["Benign", "Malignant"]
    )
    plt.title("Optimized Model")

    plt.tight_layout()
    plt.show()

    # Calculating training error for both models
    original_train_accuracy = model.score(X_train, y_train)
    optimized_train_accuracy = optimized_model.score(X_train, y_train)

    original_train_error = 1 - original_train_accuracy
    optimized_train_error = 1 - optimized_train_accuracy

    # Creating a table to display the errors
    table_data = {
        "Model": ["Original", "Optimized"],
        "Training Accuracy": [original_train_accuracy, optimized_train_accuracy],
        "Training Error": [original_train_error, optimized_train_error],
    }

    table_df = pd.DataFrame(table_data)
    table_df

    print("Evaluación en el conjunto de validación:")
    evaluate_model(model, X_val, y_val)

    print("\nEvaluación en el conjunto de prueba:")
    evaluate_model(model, X_test, y_test)

    model = train_model(X_train, y_train)
    plot_learning_curve(model, X_train, y_train, title="Learning Curves for SVM")

    classes = ["benign", "malignant"]
    plot_custom_confusion_matrix(y, model.predict(X), classes)

    model_optimized = optimize_hyperparameters(X_train, y_train)

    # Predicting the labels using the trained model
    y_pred = model.predict(X_val)

    # Plotting the confusion matrix
    plot_custom_confusion_matrix(y_val, y_pred, classes=["Benign", "Malignant"])


if __name__ == "__main__":
    main()
