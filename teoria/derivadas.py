# Derivadas
# La funcion de perdida depende de muchas variables

# ¿Por que son importantes las derivadas?
# Las redes ajustan sus pesos usando gradientes, que son derivadas parciales
# Estas permiten medir como cambia la salida con respecto a un pequeño cambio en la entrada
# Hay que optimizar la funcion de perdida mediante descenso de gradiente
# Un gradiente es un vector con todas las derivadas parciales de una funcion

# Reglas basicas
# 1. La derivada de una constante es cero
# 2. La derivada de x^n es n*x^(n-1)
# 3. La derivada de e^x es e^x
# 4. Regla del producto: (f*g)' = f'g + fg'
# 5. Regla del cociente: (f/g)' = (f'g - fg')/g^2
# 6. Regla de la cadena: d(g(f(x)))/dx = d(g(f(x)))/d(f(x)) * d(f(x))/dx
#                  otro: d[f(g(h(x)))])/dx = d[f(g(h(x)))])/d(g(h(x))) * d[g(h(x)))/d(h(x)) * d(h(x))/dx

#Ejemplo
import sympy as sp
x = sp.Symbol('x')
f = x**2 + 3*x + 5
print("Derivada de f: ", sp.diff(f, x))

g = sp.exp(x) # e^x
print("Derivada de g: ", sp.diff(g, x))

g = sp.exp(f)
print("Derivada de g: ", sp.diff(g, x))


# Gradiente
# Es un vector que contiene todas las derivadas parciales de una funcion respecto a sus variables de entrada

# Ejemplo
# f(x, y) = x^2 + y^2
# grad(f) = [df(x,y)/dx, df(x,y)/dy] = [2x, 2y]


# Regla de la cadena y su aplicacion en Backpropagation
# Permite calcular la derivada de una funcion compuesta

# Ejemplo
f = x**2 
h = sp.exp(f)
g = h
print("Derivada de la funcion compuesta: ", sp.diff(g, x))


# Descenso de gradiente
# Actualizar los pesos: w = w - L(w)
# w son los pesos, L(w) es la gradiente de la funcion de perdida respecto a los pesos

# Ejemplo
eta = 0.1 # tasa de aprendizaje
w = 2.0 # peso incial

def loss(w):
    return (w - 3)**2 # funcion de perdida simple

def grad_loss(w):
    return 2 * (w - 3) # gradiente de la funcion de perdida

for i in range(10):
    w = w - eta * grad_loss(w)
    print(f"Iteracion {i + 1}: Peso actualizado =", w)


# Ejemplos
# Single layer network
# Input: x, pesos: w, bias: b, output: y = w*x + b, funcion de perdida: L = 1/2(y - y_pred)^2
# y_pred es una funcion de w, representa el output de la red
# L es la funcion de y_pred, ya que y es constante

# dL/dw = dL/dy_pred * dy_pred/dw
#       = d(1/2y^2 - y*y_pred + 1/2y_pred^2)/dy_pred * (w*x + b)/dw
#       = (y_pred - y) * x
#       = y_pred * x - y * x

# Two layer Network
