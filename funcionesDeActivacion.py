# Funcion Escalonada (step)
# Produce valores de 0 o 1
# Ecuacion: f(x) = 1 si x >= 0, 0 si x < 0
import numpy as np
import matplotlib.pyplot as plt # libreria para graficar

def step_function(x):
    return np.where(x >= 0, 1, 0) # operador ternario

x = np.linspace(-10, 10, 100) # un arreglo de 100 numeros entre -10 y 10, inclusivo
y = step_function(x) # agarra un rango de valores de x para hacer la grafica

plt.plot(x, y) # graficar
plt.title('Funcion Escalonada') # titulo
plt.show() # mostrar la grafica


# Funcion Sigmoide
# Produce valores entre 0 y 1. Esto representa una probabilidad. Ideal para calificacion binaria
# Ecuacion: f(x) = 1 / (1 + e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)

plt.plot(x, y)
plt.title('Funcion Sigmoide')
plt.show()



# Funcion Tanh (Tangente Hiperbolica)
# Produce valores entre -1 y 1. Es similar a la sigmoide pero con un rango de valores mas amplio.
# Ecuacion: f(x) = (e^x - e^-x) / (e^x + e^-x)
def tanh(x):
    return np.tanh(x)

y = tanh(x)

plt.plot(x, y)
plt.title('Funcion Tanh')
plt.show()



# Funcion ReLU (Rectified Linear Unit)
# Produce valores entre 0 y x
# Ecuacion: f(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.title('Funcion ReLU')
plt.show()



# Funcion Leaky ReLU
# Similar a la ReLU pero con un rango de valores mas amplio.
# Ecuacion: f(x) = max(cx, x) donde c es un valor muy pequeño
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

y = leaky_relu(x)

plt.plot(x, y)
plt.title('Funcion Leaky ReLU')
plt.show()



# Funcion TELU (Trained Exponential Linear Unit)
# Ayuda a prevenir neuronas muertas asi como la leaky ReLU. Es mas estable en el entrenamiento
# Ecuacion: f(x) = x si x > 0, a(e^x - 1) si x <= 0

def telu(x, alpha=1.0): # alpha es un valor que se puede cambiar durante el entrenamiento
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

y = telu(x)

plt.plot(x, y)
plt.title('Funcion TELU')
plt.show()



# Funcion Softmax
# Utilizado en la ultima capa de una red neuronal para clasificacion multiclase.
# Entre mas grande sea el valor de x, mas probable es que sea la clase correcta.
# La suma de todas las salidas es igual a 1
# Ecuacion: f(x) = e^x / sum(e^x)
def softmax(x):
    exp_x =  np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1)

