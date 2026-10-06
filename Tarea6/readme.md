# Modelos de Clasificación: SVM, K-Nearest Neighbors y Árboles de Decisión

Este repositorio contiene el estudio, documentación y aplicación práctica de tres algoritmos fundamentales de Machine Learning. Como caso de estudio y aplicación, se utiliza el conjunto de datos Iris, el cual contiene cuatro características (longitud y ancho de sépalos y pétalos) de 50 muestras de tres especies de Iris (Iris setosa, Iris virginica e Iris versicolor)[cite: 1]. Estas medidas se utilizan frecuentemente para probar algoritmos de clasificación[cite: 1].

## 1. Support Vector Machines (SVM)
**Estudio del algoritmo:** 
Las Máquinas de Vectores de Soporte (SVM) son modelos de aprendizaje supervisado que buscan encontrar un hiperplano óptimo que separe las distintas clases de datos maximizando el margen (distancia) entre los puntos más cercanos de cada clase (conocidos como vectores de soporte).
* **Mejora en la documentación:** En su formulación lineal, SVM busca una línea recta (o hiperplano en múltiples dimensiones) rígida. El parámetro `C` controla la penalización por clasificaciones erróneas; un `C` infinito crea un modelo de clasificación de margen duro que no permite errores en el conjunto de entrenamiento[cite: 1].
* **Aplicación:** Se toma el modelo base que filtra las especies setosa y versicolor[cite: 1] y se modifica para clasificar las tres especies utilizando todo el conjunto de datos dividido en entrenamiento y prueba, evaluando su precisión general en entornos multiclasificación.

## 2. K-Nearest Neighbors (K-NN)
**Estudio del algoritmo:**
K-NN es un algoritmo basado en instancias y de tipo "perezoso" (lazy learning), lo que significa que no construye un modelo interno, sino que almacena los datos de entrenamiento y clasifica un nuevo punto de datos basándose en la clase mayoritaria de sus "K" vecinos más cercanos mediante una métrica de distancia (generalmente euclidiana).
* **Mejora en la documentación:** A diferencia de SVM, K-NN es altamente sensible a la escala de los datos, por lo que estandarizar las características es crucial. El hiperparámetro `n_neighbors` (K) debe seleccionarse cuidadosamente: un valor muy bajo genera sobreajuste frente al ruido, mientras que un valor muy alto difumina los límites de decisión.
* **Aplicación:** Se aplica el algoritmo de clasificación K-NN sobre las mismas características (longitud y ancho del pétalo) del conjunto Iris, lo que permite comparar el límite de decisión dinámico de K-NN frente al hiperplano rígido de SVM.

## 3. Árboles de Decisión (Decision Trees)
**Estudio del algoritmo:**
Los árboles de decisión modelan las decisiones dividiendo iterativamente el conjunto de datos en subconjuntos más pequeños en función de las condiciones de las características que maximizan la pureza del nodo resultante (reduciendo la impureza de Gini o la entropía).
* **Mejora en la documentación:** Los árboles de decisión son modelos de "caja blanca", lo que significa que sus reglas son fácilmente interpretables por humanos. Sin embargo, son propensos al sobreajuste si no se limitan mediante técnicas como la poda (pruning) o estableciendo una profundidad máxima (`max_depth`). No requieren escalado de características.
* **Aplicación:** Se implementa un clasificador de árbol de decisión restringiendo su profundidad para evitar el sobreajuste. Se utiliza para generar reglas claras de segmentación sobre las dimensiones de los pétalos de las flores Iris.
