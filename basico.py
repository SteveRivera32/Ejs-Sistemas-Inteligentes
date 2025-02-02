#### Tipos de Variables ####
# Cuando se asigna una variable directamente se declara, no hace falta declararla aparte
suma = 1234
y = 3.14159
str1 = "Bla Bla"
str2 = 'Bla Bla 2'
print(suma, y) # En la termina se puede poner solo el nombre de la variable para ver el valor
print (str1, str2)
# Aunque no sea necesario, podemos aclarar que tipo sera la variable para que quede mas claro en el codigo
x: int = 13018309
e: float = 2.7172
h: int = 'String' #estas anotaciones no afectan en nada, solo para documentar
# No hay limite para el tamaño de un int
num = 10398130918301983019830983019830983103801398103981309381983019380198083109381301983019831
print(num)

#### Operaciones ####
# Solo hay 2 operaciones que cambian con respecto a C++, division y exponente
x = 7
y = 2
print (x // y) #division ENTERA (no da decimales)
print (x ** y) #exponente
print (x / y) # Para hacer division REAL o con float se usa:

#### Cadenas ####
concatenado = "Hola, " + "Steve" # Concatenar cadena
repetido = "Hola" * 3 # Repetir cadena para concatenar
# Subcadenas
texto = "Python es genial"
print(texto[0:6]) # Index del inicio y hasta donde cortara (el ultimo es NO INCLUSIVO)
print(texto[-1:]) # Negativo significa contar desde el final
print(texto[:])# si no ponemos el primero es "desde el inicio" y si no ponemos el segundo es "hasta el final"
# F-strings
nombre = "Leonardo da Vinci"
edad = 67
saludo = f"Hola, mi nombre es {nombre} y tengo {edad} años"
print(saludo)
saludo = "Hola, mi nombre es " + nombre + " y tengo " + str(edad) + "años" # Como seria con concatenacion

# Boolean
es_correcto = True # Debe ir con Mayuscula

#### Listas ####
mi_lista = [] # Vacia
numeros = [10, 20, 30, 40]
numeros[1] = 100
print(numeros[1])
mixta = [1, "Hola", 3.14, True]
# existen listas mutables e inmutables (que no se pueden cambiar sus elementos). Los strings son listas inmutables.
print(numeros[1:3]) # Soportar dividir como las cadenas
# Se pueden concatenar listas y multiplicar
a = [1, 2, 3]
b = [4, 5]
print(a + b)
print (a * b)
# metodos comunes
numeros.append(50) # agregar valor al final de la lista
a.extend(b) # concatenar pero modificando 'a' en lugar de generar una nueva
a.insert(1, 300) # insertar elemento 300 en posicion 1
a.remove(30) # elimina la primera aparicion de 30
a.pop() # saca el ultimo elemento de la lista
a.pop(3) # saca el elemento en posicion 3
a.index(40) # devuelve posicion del primer elemento de 40
a.count(40) # cuenta cuantas veces aparece 40
a.sort() # ordena la lista
a.reverse() # invierta la lista
len(a) # retorna longitud de lista
# Compresion de listas (List Comprehension)
listaRango = list(range(10)) # Crear una lista con ese rango, ultimo no inclusivo
cuadrados = [x**2 for x in range(10)] # Crear una lista con una funcion
pares = [x for x in range(10) if x % 2 == 0] # Puede incluir condicion

#### Funcion sum() ####
suma = sum(pares) # Suma todos los elementos de una lista

#### Funcion zip() ####
alumnos = ["Steve", "Josue 1", "Josue 2", "Melissa"]
notas = [89, 61, 60, 95]
juntar = list(zip(alumnos, notas)) # Junta los valores de los indices en cada lista para hacer una lista de tuplas
# Las tuplas son INMUTABLES (pueden ser mas de 2)

#### Ciclo ####
for st in zip(alumnos, notas):
    print(st)
    # Los ciclos funcionan con IDENTACION

for n in range(10):
    print(n)

lista = []
for n in range(10):
    lista.append(n)

#### Funciones ####
def greet(name: str) -> None: # No retorna nada
    print(name)

greet('Elisa') # Llamar funcion

def add(a: float, b: float) -> float:
    return a + b

def imprimir_nombres(*nombres) -> None: # El * es para que reciba una tupla con parametros infinitos
    # Investigar sobre la notacion **
    print(nombres)

#### Producto Punto ####
# A . B = (a1 * b1) + (a2 * b2) + (a3 * b3)
# Podemos ver esto como una simple multiplicacion de matrices
A = [1,2,3]
B = [4,5,6]
# Implementaciones para producto punto:
AdotB = sum([A[x] * B[x] for x in range(len(A))])
AdotB = sum([x*y for (x,y) in zip(A,B)])

#### NumPy (Libreria) ####
import numpy as np # importar lib y renombrar
a = np.array([1,2,3]) # Convertir una lista de python en un arreglo de numpy
print(np.dot(a,b)) # numpy puede hacer producto punto de un solo
