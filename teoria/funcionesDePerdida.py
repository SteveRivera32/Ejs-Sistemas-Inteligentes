# Funciones de Perdida
# Es que tan cerca o lejos esta la salida de la red neuronal de la salida esperada
# Devuelve un valor numerico que representa que tan bien o mal esta la red neuronal
# de 0 a 1 (representa el porcentaje de acierto de la red neuronal)

# Tipos
# Error Cuadratico Medio (MSE - Mean Squared Error)
# Se usa para problemas de regresion, penaliza mas los errores grandes
# Ecuacion: 1/n * sumatoria((y - y_pred)^2) donde y: valor real, y_pred: valor predicho
import numpy as np

def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2) # mean = promedio

y_real = np.array([1.5, 2.0, 3.5])
y_pred = np.array([1.4, 2.1, 3.2])
print("MSE:", mse(y_real, y_pred))


# Error Absoluto Medio (MAE - Mean Absolute Error)
# Se usa para problemas de regresion, penaliza igual los errores grandes y pequeños
# Ecuacion: 1/n * sumatoria(|y - y_pred|) donde y: valor real, y_pred: valor predicho
def mae(y_real, y_pred):
    return np.mean(np.abs(y_real - y_pred))

print("MAE:", mae(y_real, y_pred))


# Entropia Cruzada Categorica (CCE - Categorical Crossentropy)
# Se usa para problemas de clasificacion, penaliza predicciones incorrectas de manera severa e igual
# Ecuacion: -sumatoria(y * log(y_pred)) donde y: valor real, y_pred: valor predicho
def cross_entropy(y_real, y_pred):
    return -np.sum(y_real * np.log(y_pred))

y_real = np.array([1, 0, 0]) # Clase real (one-hot encoding: solo una clase es 1)   
y_pred = np.array([0.7, 0.2, 0.1]) # Prediccion del modelo
print("Entropia Cruzada:", cross_entropy(y_real, y_pred))


# Entropia Cruzada Binaria (BCE - Binary Crossentropy)
# Se usa para problemas de clasificacion binaria, penaliza predicciones incorrectas de manera severa
# Ecuacion: -sumatoria(y * log(y_pred) + (1 - y) * log(1 - y_pred)) donde y: valor real, y_pred: valor
def binary_cross_entropy(y_real, y_pred):
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

y_real = np.array([1, 0, 1]) # Clase real (0 o 1)
y_pred = np.array([0.9, 0.1, 0.8]) # Prediccion del modelo
print("Binary Cross-Entropy:", binary_cross_entropy(y_real, y_pred))


# Comparacion
# MSE: regresion - penaliza mas los errores grandes
# MAE: regresion - penaliza igual los errores grandes y pequeños
# CCE: clasificacion - penaliza predicciones incorrectas de manera severa e igual
# BCE: clasificacion binaria - penaliza predicciones incorrectas de manera severa