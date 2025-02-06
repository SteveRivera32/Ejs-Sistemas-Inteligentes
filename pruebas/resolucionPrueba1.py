# Ejercicio 1
def salida_neurona(entradas, pesos, sesgo):
    valor = 0
    for x in range (len(pesos)):
        valor += entradas[x] * pesos[x]

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
    def __init__(self, cant_neuronas, cant_entradas):
        self.pesos = np.random.rand(cant_neuronas, cant_entradas) * 0.01
        self.sesgos = np.zeros.randn(cant_neuronas, 1) * 0.01

    def forward(self, datos):
        # Hacer transpuesta de pesos
        self.salidas = np.dot(self.pesos, datos) + self.sesgos