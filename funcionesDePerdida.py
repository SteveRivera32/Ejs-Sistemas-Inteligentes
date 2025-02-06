# Ej - red neuronal de 5 entradas, 2 capas ocultas (10 neuronas cada una) y 4 salidas
# Resolvera un problema de clasificacion (capas ocultas: ReLU, capa de salida: Softmax)
import numpy as np

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.rand(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos: list[float]):
        self.salida = np.dot(datos, self.pesos) + self.sesgos 

class ReLU:
    def forward(self, x: list[float]) -> None:
        self.salida = np.maximum(0, x)

class Softmax:
    def forward(self, x: list[float]) -> None:
        exp_x = np.exp(x - np.max(x))
        self.salida = exp_x / np.sum(exp_x, axis=1)

capa1 = CapaDensa(5, 10)
capa2 = CapaDensa(10, 10)
capa_salida = CapaDensa(10, 4)

relu1 = ReLU()
relu2 = ReLU()
softmax_salida = Softmax()

entradas = [1, 2, 3, 4, 5]

capa1.forward(entradas)
relu1.forward(capa1.salida)
capa2.forward(relu1.salida)
relu2.forward(capa2.salida)

capa_salida.forward(relu2.salida)
softmax_salida.forward(capa_salida.salida)
print(softmax_salida.salida)