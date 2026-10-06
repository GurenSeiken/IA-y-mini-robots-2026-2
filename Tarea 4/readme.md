# Tarea 4: Programación Genética (PG)

¡Sorpresa con la Tarea 4! 🎉
En esta carpeta se han desarrollado los puntos 2 y 4 de los ejercicios de Programación Genética (PG), haciendo uso de cuadernos de Python para su ejecución y visualización interactiva.

## Estructura del Proyecto

- `Notebook_Punto2_Galletas.ipynb`: Cuaderno de Jupyter que implementa el algoritmo genético para el robot repartidor de galletas.
- `Notebook_Punto4_Inventado.ipynb`: Cuaderno de Jupyter que implementa un problema inventado de Regresión Simbólica.
- `create_notebooks.py`: (Opcional) Script usado para inicializar los cuadernos desde cero.

---

## Decisiones de Diseño y Arquitectura

Para ambos problemas, hemos utilizado la librería **DEAP** (Distributed Evolutionary Algorithms in Python), ya que es un estándar robusto y flexible para desarrollar algoritmos evolutivos en Python.

### Punto 2: Robot Repartidor de Galletas 🍪
El problema plantea un robot que debe moverse por una sala cuadrada entregando galletas a un grupo de ingenieros. Para solucionarlo usando Programación Genética, modelamos el comportamiento del robot como un árbol de decisiones (sintaxis tipo LISP).

**Diseño del Entorno (Simulador):**
- **Sala:** Cuadrícula (grid) de 10x10.
- **Ingenieros:** 15 ingenieros ubicados aleatoriamente en la sala.
- **Robot:** Comienza en (0,0) mirando hacia la derecha.

**Conjuntos de la PG:**
1. **Conjunto de Terminales (Acciones básicas del robot):**
   - `Avanzar`: Mueve al robot una casilla hacia adelante. Si la casilla tiene un ingeniero, entrega automáticamente la galleta.
   - `GirarIzq`: Gira el robot 90° a la izquierda sin avanzar.
   - `GirarDer`: Gira el robot 90° a la derecha sin avanzar.
2. **Conjunto de Funciones (Control de flujo):**
   - `prog2`: Ejecuta secuencialmente 2 instrucciones.
   - `prog3`: Ejecuta secuencialmente 3 instrucciones.
   - `if_engineer_ahead`: Función condicional (sensor). Evalúa si hay un ingeniero en la casilla inmediatamente al frente; de ser así, ejecuta su primer argumento, sino, ejecuta el segundo.
3. **Función de Aptitud (Fitness):**
   - El objetivo principal es maximizar las galletas entregadas.
   - *Fitness* = Número de galletas entregadas (máximo 15). Detenemos la simulación a los 200 pasos para evitar bucles infinitos. Se busca **maximizar** este valor.

*Justificación:* Este enfoque es similar al clásico problema "Santa Fe Ant Trail". Las funciones de control de flujo (`prog2`, `prog3`, `if`) permiten que el algoritmo genético ensamble bucles implícitos y rutinas de exploración inteligentes, mientras que las terminales aseguran la acción directa.

---

### Punto 4: Problema Inventado - Predicción de Consumo Energético de Mini-Robots 🔋
Para el cuarto punto, hemos diseñado un problema de **Regresión Simbólica**. Imagina que estamos diseñando un nuevo mini-robot y queremos obtener la ecuación física que describe su consumo energético basándonos en variables del entorno y del diseño.

**El Problema:**
Encontrar una función $E = f(v, p, i)$ que determine el consumo energético en función de la velocidad ($v$), peso ($p$) y la inclinación del terreno ($i$).

**Conjuntos de la PG:**
1. **Conjunto de Terminales (Variables y constantes):**
   - Variables de entrada: `v` (Velocidad), `p` (Peso), `i` (Inclinación).
   - Constantes numéricas y efímeras para que el algoritmo pueda ensamblar los coeficientes de la fórmula.
2. **Conjunto de Funciones (Operadores matemáticos):**
   - Suma (`+`), Resta (`-`), Multiplicación (`*`), y División Protegida (`/` segura contra divisiones por cero).
3. **Función de Aptitud (Fitness):**
   - El objetivo es minimizar el error de las predicciones frente a un dataset real recolectado en pruebas de campo.
   - *Fitness* = Error Cuadrático Medio (MSE) entre el valor que da el árbol genético y el valor real de la base de datos de prueba. Se busca **minimizar** este valor.

*Justificación:* La Regresión Simbólica es uno de los usos más fascinantes de la PG. A diferencia del Deep Learning, donde obtendríamos una red neuronal ("caja negra"), la PG nos entrega una fórmula matemática legible por humanos, lo cual es muy útil para ingeniería de diseño de robots.

## ¿Cómo ejecutar los notebooks?
Asegúrate de instalar los requerimientos dentro de los propios notebooks (`!pip install deap matplotlib numpy`) y corre todas las celdas de manera secuencial.
