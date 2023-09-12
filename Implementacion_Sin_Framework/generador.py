# ----------------------------------------------------------
# Archivo, generador de datos al azar.
#
# Date: 27-Aug-2022
# Authors:
#           A01379299 Ricardo Ramírez Condado
#
# Fecha de creación: 24/08/2022
# Última actualización: 27/08/2022
# ----------------------------------------------------------


import random
import csv


# Generar datos aleatorios
def generate_random_data(num_data):
    data = []
    for _ in range(num_data):
        tumor_size = round(random.uniform(0.5, 5.0), 2)
        cell_uniformity = random.randint(1, 10)
        cell_adhesion = random.randint(1, 10)
        cell_size = random.randint(1, 10)
        nuclear_nudeness = random.randint(1, 10)
        label = random.randint(0, 1)
        data.append(
            [
                tumor_size,
                cell_uniformity,
                cell_adhesion,
                cell_size,
                nuclear_nudeness,
                label,
            ]
        )
    return data


# Guardar datos en un archivo CSV
def save_data_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "Tumor_Size",
                "Cell_Uniformity",
                "Cell_Adhesion",
                "Cell_Size",
                "Nuclear_Nudeness",
                "Label",
            ]
        )
        csv_writer.writerows(data)


# Generar 1000 datos aleatorios y guardar en CSV
num_data = 1000
random_data = generate_random_data(num_data)
save_data_to_csv(random_data, "random_data.csv")

print("Datos aleatorios generados y guardados en 'random_data.csv'")
