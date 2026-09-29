# Punto 1: Modelo de Red Neuronal con Dos Capas Ocultas

Este directorio contiene la solución al Punto 1, donde se implementa una Red Neuronal Artificial para realizar predicciones sobre los datos de un experimento.

## 1. Elección de los Datos (El Experimento)

Para este ejercicio, hemos escogido el **Wine Recognition Dataset** (Conjunto de datos de reconocimiento de vinos), disponible de forma nativa en la librería `scikit-learn`.

**¿Por qué escogimos estos datos?**
- **Representa un experimento real:** Los datos son el resultado de un análisis químico real (un experimento de laboratorio) realizado a vinos cultivados en la misma región en Italia, pero derivados de tres cultivares (tipos de cepas) diferentes.
- **Limpieza y Estructura:** Es una tabla de datos muy bien estructurada, con 178 muestras y 13 características químicas cuantitativas (como niveles de alcohol, ácido málico, magnesio, fenoles, color, etc.).
- **Ideal para Redes Neuronales:** Al tener múltiples características numéricas con diferentes escalas, es el escenario perfecto para aplicar preprocesamiento y entrenar un modelo de clasificación con redes neuronales.

## 2. Explicación del Modelo (Paso a Paso)

El código fuente principal se encuentra en `modelo.py`. A continuación se explica todo el flujo de trabajo y lo que hace el código paso a paso:

### Paso 1: Carga de Datos
El primer paso del script importa el dataset del experimento usando `load_wine()`. Esto nos entrega una matriz `X` con las características químicas (nuestros datos de entrada) y un vector `y` con las etiquetas (la clase de vino que es: tipo 0, 1 o 2).

### Paso 2: Preprocesamiento (División y Escalamiento)
- **División:** Usamos `train_test_split` para dividir aleatoriamente la tabla de datos en dos partes: el **80%** se usará para entrenar o "enseñar" a la red neuronal, y el **20%** se guardará para probarla después y ver qué tan bien aprendió.
- **Escalamiento (Fundamental):** Las Redes Neuronales son altamente sensibles a la magnitud de los datos. Si el nivel de alcohol está en el rango de 11 a 14, pero el magnesio está entre 70 y 162, el modelo le dará más peso erróneamente al magnesio solo por ser un número más grande. Para evitar esto, usamos `StandardScaler`, que estandariza todas las características para que tengan una media de 0 y una desviación estándar de 1.

### Paso 3: Construcción de la Red Neuronal
Aquí creamos el modelo. La instrucción pide específicamente **"una red neuronal de dos capas ocultas"**.
Para ello, utilizamos `MLPClassifier` (Multi-Layer Perceptron) configurado de la siguiente manera:
- `hidden_layer_sizes=(16, 8)`: Esto crea exactamente dos capas ocultas. La primera capa oculta tendrá 16 neuronas, y la segunda tendrá 8 neuronas.
- `activation='relu'`: Usamos la función de activación ReLU (Rectified Linear Unit), que es el estándar en la industria por su eficiencia y rapidez para evitar el desvanecimiento del gradiente.
- `solver='adam'`: Es el algoritmo de optimización que ajustará los pesos de las neuronas para reducir el error durante el aprendizaje.

### Paso 4: Entrenamiento
Ejecutamos `modelo_nn.fit(X_train_scaled, y_train)`. Durante este proceso, el modelo recibe los datos de entrenamiento y ajusta sus parámetros internos miles de veces usando un proceso llamado retropropagación (backpropagation) hasta encontrar patrones entre los componentes químicos y el tipo de vino.

### Paso 5: Alimentar datos para hacer una Predicción
Finalmente, cumplimos con la última parte de la instrucción: *"Dele datos a ese modelo para hacer una predicción"*.
- Tomamos un dato del mundo real (simulado a partir de nuestra partición de prueba).
- Le mostramos al usuario cuáles son las medidas químicas de este vino.
- **Escalamos** ese dato usando las mismas métricas que el modelo aprendió durante el entrenamiento.
- Llamamos a la función `predict()`. El modelo toma estas métricas, las pasa por sus dos capas ocultas de neuronas, y devuelve su decisión sobre a qué cultivar de vino pertenece.

## 3. ¿Cómo ejecutar el código?

Si deseas correr el modelo tú mismo y ver la predicción en tiempo real, simplemente ejecuta el siguiente comando en la terminal (asegúrate de tener instalada la librería `scikit-learn`):

```bash
pip install scikit-learn
python "modelo.py"
```

## 4. Resultados Obtenidos

Al ejecutar el modelo de red neuronal de dos capas ocultas sobre este experimento, se obtuvieron los siguientes resultados formales:

- **Precisión (Accuracy):** El modelo logra una precisión del **100.00%** al ser evaluado con el conjunto de prueba (que corresponde al 20% de los datos que el modelo jamás había visto en su entrenamiento). Esto indica que la red neuronal logró extraer y aprender los patrones químicos con total exactitud.
- **Predicción con un nuevo dato:** Para demostrar el punto *"Dele datos a ese modelo para hacer una predicción"*, se aisló un registro aleatorio de vino con características específicas (por ejemplo, `alcohol: 13.64`, `magnesium: 116.0`, etc.). 
- **Conclusión de la Predicción:** El modelo predijo que el cultivo correspondiente era la clase `class_0`. Al contrastarlo con la clase real del registro, el resultado fue exactamente el mismo. Esto nos demuestra empíricamente que la red neuronal no solo aprendió, sino que es capaz de generalizar y predecir correctamente sobre nuevos datos que se le presenten en el futuro.

---

# Punto 5 (Libre): Diagnóstico Médico con TensorFlow

Para este punto de libre elección, queríamos un modelo que, a pesar de ser muy veloz y liviano, tuviera una aplicación gigantesca en el mundo real. Por ello, incursionamos en la **Inteligencia Artificial aplicada a la Medicina**.

## 1. El Enunciado Propuesto

*Utilizando TensorFlow/Keras y datos médicos reales, construya una Red Neuronal capaz de predecir si un tumor de cáncer de mama es 'Maligno' o 'Benigno' basándose en características celulares. Entrene el modelo, analice su precisión y realice una predicción sobre un caso clínico de prueba.*

## 2. Archivos y Ejecución

La solución a este punto se encuentra en el archivo interactivo: `punto5.ipynb`. Se diseñó como un cuadernillo de Colab (Jupyter Notebook) para poder visualizar cada paso médico con claridad y ejecutarlo en cuestión de milisegundos sin consumir memoria excesiva.

### ¿Qué contiene el cuadernillo?
1. **Carga del Dataset Médico:** Descargamos el *Breast Cancer Wisconsin Dataset* de `scikit-learn`. Contiene el historial de cientos de pacientes con 30 mediciones exactas del núcleo de sus células (radio, textura, área, etc.).
2. **Preprocesamiento:** Estandarizamos los datos usando `StandardScaler`. Al igual que en el Punto 1, esto es vital para evitar sesgos numéricos.
3. **Construcción del Modelo (TensorFlow):**
   - Utilizamos la API de Keras para construir una red neuronal secuencial.
   - Capas `Dense`: Posee dos capas ocultas (16 y 8 neuronas respectivamente) con activación ReLU.
   - Capa de Salida: Una sola neurona con activación Sigmoid, que arroja la probabilidad (de 0 a 1) de que el tumor sea maligno o benigno.
4. **Entrenamiento Relámpago:** Entrenamos el modelo durante 30 épocas. Dado lo ligero de la base de datos, ¡esto toma literalmente milisegundos en completarse!
5. **Predicción Interactiva:** Simulamos la llegada de un paciente al hospital aislando un dato del conjunto de prueba. Pasamos sus mediciones celulares por la red neuronal ya entrenada y le pedimos un diagnóstico automático, comparándolo de inmediato con el veredicto real de laboratorio.

Este punto es una excelente muestra de lo que las redes neuronales pueden hacer de manera eficiente. Sin necesidad de descargar gigas de imágenes o procesar grandes textos, podemos lograr que TensorFlow salve vidas detectando patrones matemáticos en análisis de laboratorio.

### 3. Resultados Obtenidos
Tras ejecutar el cuadernillo, la red neuronal obtuvo los siguientes resultados:
- **Precisión (Accuracy):** Alcanzó un impresionante **97.37%** de precisión al diagnosticar el conjunto de datos de prueba (casos médicos que nunca había visto).
- **Prueba Clínica Real:** Al tomar un paciente al azar, el modelo calculó una probabilidad del `94.78%` (0.9478) de que el tumor fuera Benigno. 
- **Veredicto:** El diagnóstico de la Inteligencia Artificial fue **Benigno**, lo cual coincidió perfectamente con los resultados de laboratorio reales del paciente.
