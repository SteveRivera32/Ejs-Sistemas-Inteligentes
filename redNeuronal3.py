# Ej - Agregar las funciones de activacion dentro de CapaDensa
import numpy as np

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int, activacion):
        self.pesos = np.random.rand(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))
        self.activacion = activacion

    def forward(self, datos: list[float]):
        salidaTemp = np.dot(datos, self.pesos) + self.sesgos 
        self.salida = self.activacion.forward(salidaTemp)

class ReLU:
    def forward(self, x: list[float]) -> None:
        return np.maximum(0, x)

class Softmax:
    def forward(self, x: list[float]) -> None:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1)

relu1 = ReLU()
relu2 = ReLU()
softmax = Softmax()

capa1 = CapaDensa(5, 10, relu1)
capa2 = CapaDensa(10, 10, relu2)
capa_salida = CapaDensa(10, 4, softmax)

entradas = [1, 2, 3, 4, 5]

capa1.forward(entradas)
capa2.forward(capa1.salida)
capa_salida.forward(capa2.salida)

print(capa_salida.salida)
                                                                                             