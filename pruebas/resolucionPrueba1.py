# Ejercicio 1
# Resolucion ing
def salida_neurona(entradas, pesos, sesgo):
    valor = 0
    for x in range (len(pesos)):
        valor += entradas[x] * pesos[x]

    return valor + sesgo

# Resolucion mia
def salida_neurona(entradas, pesos, sesgo):
    valor = 0
    for (entrada, peso) in zip(entradas, pesos):
        valor += entrada * peso

    return valor + sesgo



# Ejercicio 2
def salida_capa(entradas, pesos, sesgos):
    salidas = []
    for n_sesgo, n_pesos in zip(sesgos, pesos):
        salidas.append(sum(i*w for i, w in zip(entradas, n_pesos)) + n_sesgo)

    return salidas

# Ejercicio 3
import numpy as np

class CapaDensa:
    def __init__(self, cant_entradas, cant_neuronas):
        self.pesos = np.random.rand(cant_neuronas, cant_entradas) * 0.01
        self.sesgos = np.random.rand(1, cant_neuronas) * 0.01

    def forward(self, entradas):
        self.salida = np.dot(self.pesos, entradas) + self.sesgos.T