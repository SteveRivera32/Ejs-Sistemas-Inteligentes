# Capa Densa
# Una capa densa es una capa donde todas las entradas estan conectadas a todas las salidas de la capa anterior
# Implementar capa
import numpy as np

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int): # "entradas" es la cantidad de entradas
        self.pesos = np.random.rand(neuronas, entradas) * 0.01 # rand crea una matriz de numeros aleatorios
        self.sesgos = np.zeros((1, neuronas)) # Crear un arreglo de ceros de 1 fila y "neuronas" columnas
        # el primer parametro son las filas y el segundo las columnas

    def forward(self, datos: list[float]): # "datos" es un arreglo de entradas a la capa
        self.salida = np.dot(datos, self.pesos) + self.sesgos # en vez de .dot se puede usar .matmul


# Ejemplo de uso
# Se puede activar una terminal en donde este el archivo y usar "import capas as c"
capa1 = CapaDensa(4, 3) # Crear una capa con 4 entradas y 3 neuronas
capa1.forward([[1, 2, 3, 2.5]])
print(capa1.salida)

capa2 = CapaDensa(capa1.salida.len(), 3) # Crear una capa con 3 entradas (cantidad de salidas de capa1) y 3 neuronas
capa2.forward(capa1.salida)
print(capa2.salida)

# Manejo de lotes (batches)
# Un batch es un conjunto de datos que se procesan al mismo tiempo
datos = np.array([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])
# al hacer el forward, como los sesgos no tendra el mismo tamaño, se debe duplicar el arreglo en varias filas

#Ejemplo
capa1.forward(datos) # esto funciona porque numpy ya hace BROADCASTING, que es duplicar el arreglo para que tenga el mismo tamaño
capa2.forward(capa1.salida)

#Ejemplo: hacer red neuronal de 10 entradas y 1 salida, con 2 capas ocultas
capa1 = CapaDensa(10, 5)
capa2 = CapaDensa(5, 5)
capaSalida = CapaDensa(5, 1)