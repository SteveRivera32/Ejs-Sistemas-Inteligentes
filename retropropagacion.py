# Retroprograpacion (Backpropagation)
# Es un algoritmo que ajusta los pesos y sesgos hacia atras para minimizar la funcion de perdida
# Utiliza la regla de la cadena para calcular derivadas de funciones compuestas

 # Vision general del proceso
# 1. Forward - calcular la salida de la red neuronal
# 2. Calcular la perdida - comparar la salida con la salida esperada
# 3. Retropropagacion - calcular gradientes con la regla de la cadena
# 4. Actualizar de parametros - ajustar pesos y sesgos

# Ejemplo de neurona con ReLU
x = [1.0, -2.0, 3.0] # Entradas
w = [-3.0, -1.0, 2.0] # Pesos
b = 1.0 # Sesgo

z = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b # Suma ponderada
y = max(z, 0) # Funcion de activacion ReLU

# Ejemplo de retropropagacion
dReLU_dz = 1.0 if z > 0 else 0.0 # Derivada de ReLU

# Si la salida de ReLU es negativa, la derivada se vuelve 0
# y por lo tanto, las gradientes de los pesos y sesgos se vuelven 0
# NOTA: esto significa que no se hara ningun ajuste a los pesos y sesgos

# Si la salida de ReLU es positiva, la derivada se vuelve 1
# y por lo tanto, las gradientes de los pesos seran las entradas, y la gradiente de entrada seran los peso
# porque la derevada de la suma ponderada con respecto a la entrada es el peso multiplicandolo y viceversa


# Volvemos con el ejemplo
# Supongamos un gradiente entrante de dvalue = 1.0
dvalue = 1.0
dReLU_dz = dvalue * (1.0 if z > 0 else 0.0) # = 1.0 * 1.0

# Gradiente parcial por cada peso
dw = [0.0, 0.0, 0.0]
dw[0] = x[0] * dReLU_dz
dw[1] = x[1] * dReLU_dz
dw[2] = x[2] * dReLU_dz

learning_rate = 0.1 # Tasa de aprendizaje
# w[i] = w[i] - learning_rate * dw[i] # Actualizacion de pesos



# Ejemplo en una capa completa
# Entradas (batch con 2 muestras y 3 caracteristicas)
X = [ [1.0, -2.0, 3.0],
      [0.5, 1.0, -1.5] ]

# 2 neuronas
W = [ [0.2, -0.4],
      [0.8, 0.1],
      [-0.5, 0.9] ]

b = [ [0.5, -0.2] ]

# Forward
import numpy as np

X = np.array(X)
W = np.array(W)
b = np.array(b)

Z = np.dot(X, W) + b # Funcion de tranferencia
A = np.maximum(Z, 0) # ReLU

# Backward
# Los dvalues que son la gradiente de entradas de la capa siguiente
dvalues = np.array([[1.0, 1.0],
                    [1.0, 1.0]])
# tiene que tener la misma dimension que la salida de la capa

dinputs = np.dot(dvalues, W.T) # Gradientes de las entradas (dim de entrada)
dweights = np.dot(X.T, dvalues) # Gradientes de los pesos (dim de pesos)
dbiases = np.sum(dvalues, axis=0, keepdims=True) # Gradientes de los sesgos (dim de sesgos)

# Actualizar pesos y sesgos
learning_rate = 0.01
W -= learning_rate * dweights
b -= learning_rate * dbiases


# Perdida de entropia categorica cruzada y softmax
# Derivada de softmax
# Esta derivada es complicada porque cada elemento de la salida depende de todos los elementos de la entrada
# Al sacar la derivada de cada salida con respecto a cada entrada, se obtiene una matriz jacobiana
# Caso 1 (i = j) - los numeros de la diagonal en la matriz
# dS_i / dz_j = S_i * (1 - S_i)

# Caso 2 (i != j) - los que no esten en la diagonal
# dS_i / dz_j = - S_i * S_j

# Ejemplo 
z = np.array([1, 2, 3, 4])
S = np.array([0.032, 0.087, 0.2369, 0.6439]) # aplicar softmax
matriz = np.array([[0.032 * (1 - 0.032), -0.032 * 0.087, -0.032 * 0.2369, -0.032 * 0.6439],
                    [-0.087 * 0.032, 0.087 * (1 - 0.087), -0.087 * 0.2369, -0.087 * 0.6439],
                    [-0.2369 * 0.032, -0.2369 * 0.087, 0.2369 * (1 - 0.2369), -0.2369 * 0.6439],
                    [-0.6439 * 0.032, -0.6439 * 0.087, -0.6439 * 0.2369, 0.6439 * (1 - 0.6439)]])

def softmax(x):
    exp_x =  np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def derivada_softmax(x):
    S = softmax(x)
    jacobiana = np.zeros((len(S), len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            if i == j:
                jacobiana[i][j] = S[i] * (1 - S[j])
            else:
                jacobiana[i][j] = -S[i] * S[j]
    return jacobiana

matriz = derivada_softmax(z)
print(matriz)