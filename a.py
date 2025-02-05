# Este archivo es para hacer pruebas de codigo
# Ej 1 - escribir codigo para calcular la salida de una capa con 2 neuronas y 3 entradas
import numpy as np

entradas = [1, 2, 3]
pesos = [
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
]
sesgos = [2, 3]

salidas = np.dot(pesos, entradas) + sesgos
print(salidas) 