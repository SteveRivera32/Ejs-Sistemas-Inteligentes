# Neuronas
# Formula para funcion de transferencia: salida = sum(entradas * pesos) + sesgo
# Implementacion basica de una neurona
entradas = [1, 2, 3]
pesos = [0.2, 0.8, -0.5]
sesgo = 2

salida = (entradas[0] * pesos[0] +
          entradas[1] * pesos[1] + 
          entradas[2] * pesos[2] +
          sesgo)
print(salida)

# Implementacion basica de una capa (conjunto de neuronas)
entradas = [1,2,3,2.5]

pesos1 = [0.2, 0.8, -0.5, 1]
pesos2 = [0.5, -0.91, 0.26, -0.5]
pesos3 = [-0.26, -0.27, 0.17, 0.87]
sesgos = [2, 3, 0.5]

salidas = [
    sum ([i * w for i, w in zip(entradas, pesos1)]) + sesgos[0],
    sum ([i * w for i, w in zip(entradas, pesos2)]) + sesgos[1],
    sum ([i * w for i, w in zip(entradas, pesos3)]) + sesgos[2],
]
print(salidas)

# Simplifica capa con ciclos
entradas = [1, 2, 3, 2.5]
pesos_capa = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
sesgos_capa = [2, 3, 0.5]

salidas = []
for pesos, sesgo in zip (pesos_capa, sesgos_capa):
    salida = sum([i * w for i, w in zip(entradas, pesos)]) + sesgo
    salidas.append(salida)

print(salidas)

# Implementacion de Numpy
import numpy as np

entradas = [1, 2, 3, 2.5]
pesos = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
sesgos = [2, 3, 0.5]

salidas = np.dot(pesos, entradas) + sesgos
print(salidas)

#Ej 1 - escribir codigo para calcular la salida de una capa con 2 neuronas y 3 entradas
import numpy as np

entradas = [1, 2, 3]
pesos = [
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
]
sesgos = [2, 3]

salidas = np.dot(pesos, entradas) + sesgos
print(salidas)