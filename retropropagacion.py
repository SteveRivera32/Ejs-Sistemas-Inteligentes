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
dReLU_dz = 1.0 if z > 0 else 0.0 # Derivada de ReLU para las entradas
