# Optimizadores
# Algoritmos que se encargan de minimizar la perdida
# Objetivo: encontrar el minimo global de la funcion de perdida
# Base: Descenso del Gradiente Estocastico (SGD - Stochastic Gradient Descent)


# Descenso del Gradiente Estocastico (SGD - Stochastic Gradient Descent)
pesos -= tasa_aprendizaje * gradientes

# Variantes
# - SGD por lotes (dataset completo)
# - SGD por mini-lotes
# - SGD estocastico (1 dato a la vez)

class Optimizer_SGD:
    def update_params(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases
# Problemas: se estanca en minimos locales

# Tasa de aprendizaje (learning rate):
# - Demasiado alta: sobrepaso, divergencia
# - Demasiado baja: entrenamiento lento, minimos locales

# Decaimiento de LR:
lr_actual = lr_inicial / (1 + decaimiento * iteracion)

# SGD con momentum
# Acumula gradientes pasados para superar minimos locales
actualizacion = momentum * actualizacion_anterior - lr * gradiente
actualizacion_pesos = (momento * momento_previo) - (lr * gradiente)


# Adagrad (Adaptive Gradient Algorithm)
# Normalizar actualizaciones: con actualizaciones grandes, se ajustan mas lento, y viceversa
# Los parametros menos frecuentes se ajustan mas rapido
cache += gradiente ** 2
parametro -= tasa_aprendizaje * gradiente / (np.sqrt(cache) + epsilon) # epsilon=1e-7 evita division por 0


# RMSprop (Root Mean Square Propagation)
# Promedio movil de gradientes al cuadrado (suaviza historial)
# Solo cambia como calcula el cache
cache = rho * cache + (1 - rho) * gradiente ** 2
parametro -= tasa_aprendizaje * gradiente / (np.sqrt(cache) + epsilon)


# Adam (Adaptive Moment Estimation)
# Combinacion de Adagrad y RMSprop
# Incluye un mecanismo de correccion de sesgo para mejorar las primeras iteraciones
# beta1 = 0.9, beta2 = 0.999
momentum = beta1 * momentum + (1 - beta1) * gradiente
cache = beta2 * cache + (1 - beta2) * gradiente ** 2
momento_corregido = momentum / (1 - beta1 ** iteracion)
cache_corregido = cache / (1 - beta2 ** iteracion)
parametro -= tasa_aprendizaje * momento_corregido / (np.sqrt(cache_corregido) + epsilon)

# TAREA: explorar variantes avanzadas
# AMSGrad: Adam with maximum squared gradient
# Nadam: Nesterov-accelerated Adaptive Moment Estimation