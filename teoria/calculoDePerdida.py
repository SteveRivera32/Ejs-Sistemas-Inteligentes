# Calculo de la perdida de la red
# La funcion de perdida nos sirve para medir tambien la confianza y no solo la precision
# Ej
# Prediccion 1: [0.22, 0.6, 0.18] - Argmax: 1
# Prediccion 2: [0.32, 0.36, 0.32] - Argmax: 1
# La primera prediccion es mas confiable que la segunda

# Entropia Cruzada Categorica (CCE - Categorical Crossentropy)
# Formula: -sum(y * log(y_pred))
# La formula se puede simplificar en problemas categoricos donde se usa one-hot encoding
# Formula simplificada: -log(y_pred[clase_correcta])
# Ej
# - Salida softmax: [0.1, 0.2, 0.7]
# - Etiqueta verdad: [0, 0, 1]
# - Pérdida: -log(0.7) = 0.3567


# Manejo de casos extremos
# Si y_pred = 0, la perdida es infinita: -log(0) = infinito
# Para evitarlo se cambia el 0 por un valor muy pequeño (1e-7)
# y al mismo tiempo, debemos ajustar los demas valores para que sigan sumando 1
# Ej
import numpy as np

# Salida softmax y etiquetas verdaderas
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.5, 0.4],
                             [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1]) # Etiquetas dispersas

# Recortar predicciones
y_pred_clipped = np.clip(softmax_outputs, 1e-7, 1 - 1e-7) # Reemplaza 0 por 1e-7 y 1 por 1 - 1e-7
                                                          # en el arreglo mandado

# Calcular perdida
correct_confidences = y_pred_clipped[range(len(softmax_outputs)), class_targets] # hace un arreglo de los valores correctos
negative_log_likelihoods = -np.log(correct_confidences) 
loss = np.mean(negative_log_likelihoods) # promedio de la perdida de todo el batch

print("Perdida:", loss)


# Calculo de la precision
predictions = np.argmax(softmax_outputs, axis=1) # argmax devuelve indice del valor mas grande
accuracy = np.mean(predictions == class_targets) # Promedio de aciertos
print("Precision:", accuracy)