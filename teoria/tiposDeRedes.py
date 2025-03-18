# Tipos de Redes Neuronales
# Redes Neuronales Convolucionales (CNN)
# - Calificar imagenes, videos, reconocimiento facial, etc.
# - Extraen patrones usando kernles o filtros (kernel: matriz pequeña que se desplaza multiplicando elementos locales)
# - Capas: convolucionales, de agrupacion (pooling), totalmente conectadas
# - Ejemplo: AlexNet, VGG, GoogleNet, ResNet

# Redes Neuronales Recurrentes (RNN)
# - Diseñadas para datos secuenciales (texto, audio, video, series temporales)
# - Manejan memoria para conservar informacion previa
# Variantes:
# LSTM (Long Short-Term Memory): memoria a corto y largo plazo
# - Tres puertas: olvidar, entrada, salida
# - Aplicaciones: traduccion automatica, reconocimiento de voz
# GRU (Gated Recurrent Unit): simplificacion de LSTM
# - Dos puertas: actualizacion, reinicio
# - Mas eficiente, menos parametros

# Transformers
# - Basados en mecanismos de autoatencion
# - Self-attention: permite evaluar simultaneamente todas las posiciones en la entrada para determinar relevancia
# - Redes feed-forward: capas densas simples aplicadas en cada posicion
# - Codificacion posicional: vectores que representan la posicion relativa de elementos para mantener la secuencia
# - Aplicaciones: ChatGPT, generacion de texto, traduccion, clasificacion de imagenes
# 
# Autoatencion en Transformers
# Oracion: "El gato se subio al tejado"
# 1. Convertir palabras a vectores
# 2. Crear consultas (Q), claves (K) y valores (V)
#    La consulta de una palabra se compara con las claves de todas las palabras
#    Esto genera puntuaciones ("se" tendra alta puntuacion con "gato")
# Los transformers no procesan palabras en orden (codificacion posicional)