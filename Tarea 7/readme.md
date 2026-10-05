# Informe de Diseño: Tarea 7 (RAG y Agentes)

Este documento detalla y justifica las decisiones técnicas y arquitectónicas tomadas para la implementación de los chatbots RAG y el Agente de Mantenimiento Industrial.

## 1. Selección del Modelo de Lenguaje e Infraestructura
**Decisión:** Utilizar el modelo **Gemma 2 2B** (referenciado como gemma4 e2b) desplegado a través del servidor local de **LM Studio**.
**Justificación:**
- **Limitación de Hardware:** El desarrollo se llevó a cabo en una máquina con una GPU RTX 3050 de 4GB de VRAM. Modelos estándar de 7B-8B parámetros exceden este límite, lo que causa cuellos de botella severos al usar la memoria RAM del sistema. El modelo de 2B parámetros garantiza fluidez, respuestas rápidas y cero cuelgues.
- **Interoperabilidad (LM Studio):** Se aprovechó la capacidad de LM Studio para levantar un servidor local compatible con la API de OpenAI. Esto nos permitió utilizar la clase estándar `ChatOpenAI` de LangChain apuntando al `localhost`, logrando alta eficiencia sin consumir créditos en la nube.

## 2. Arquitectura del Framework (LangChain & LCEL)
**Decisión:** Uso de LCEL (LangChain Expression Language) para orquestar el flujo de datos.
**Justificación:**
Durante el desarrollo se optó por no usar componentes antiguos (como `RetrievalQA` o dependencias de `langchain.chains`) debido a que las actualizaciones recientes de LangChain han deprecado estas librerías, generando errores de compatibilidad de módulos. La sintaxis LCEL es mucho más moderna, declarativa y permite ver explícitamente el flujo de datos: `Recuperación (Retriever) -> Formateo -> Prompt -> LLM -> Parser`.

## 3. Decisiones sobre el RAG (Puntos 2 y 3)
**Decisión:** Emplear `ChromaDB` como base de datos vectorial, `HuggingFaceEmbeddings` (modelo `all-MiniLM-L6-v2`) y archivos `.txt`.
**Justificación:**
- **Embeddings:** El modelo `all-MiniLM` es sumamente ligero y eficiente; es ideal para hacer la transformación matemática de los textos de forma local sin cargar la CPU/GPU innecesariamente.
- **ChromaDB:** Permite guardar la información vectorizada directamente en carpetas locales, asegurando persistencia entre ejecuciones sin configurar servicios como Docker o bases de datos remotas.
- **Formato Texto:** Para evitar complicaciones de extracción de texto en Windows y posibles errores de codificación (`cp1252` o librerías de PDF que fallan silenciosamente), se diseñaron los manuales de prueba directamente en formato texto plano forzando la lectura en UTF-8. 

## 4. Agente de Mantenimiento (Punto 4)
**Decisión:** Uso de un agente con soporte para invocación de herramientas (Tool Calling) y bases de datos simuladas en memoria (diccionarios y listas en Python).
**Justificación:**
- Se crearon 4 funciones aisladas mediante el decorador `@tool`: `buscar_repuesto`, `consultar_almacen`, `generar_orden_trabajo` y `actualizar_historial_fallas`.
- **Eficacia:** Separar la lógica de negocio (Python) de la inteligencia del modelo (LLM) es una buena práctica. Al usar modelos de menor escala como el de 2B, pedirles que generen lógicas complejas en texto suele causar alucinaciones. En cambio, al darles herramientas bien documentadas, el modelo solo toma la "decisión" de qué herramienta llamar y con qué parámetros, mientras que Python realiza la actualización del "historial" o del "almacén" con absoluta precisión computacional.
