"""
Punto 2: Verdadera Democracia - Reparto de Poder Político mediante Algoritmos Genéticos (AGs)
Asignatura: Inteligencia Artificial y Mini-robots
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Fijar semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# 1. DEFINICIÓN DEL ESCENARIO POLÍTICO
# ==============================================================================

PARTIDOS = [
    "Partido Renovación Democrática (PRD)",
    "Alianza Social Progresista (ASP)",
    "Fuerza Conservadora Unida (FCU)",
    "Movimiento Centro Plural (MCP)",
    "Coalición Verde & Regional (CVR)"
]

NUM_PARTIDOS = len(PARTIDOS)
TOTAL_CURULES = 50
TOTAL_ENTIDADES = 50

# Lista de 50 entidades estatales representativas
NOMBRES_ENTIDADES = [
    "Ministerio de Hacienda y Crédito Público",
    "Ministerio del Interior",
    "Ministerio de Defensa Nacional",
    "Ministerio de Justicia y del Derecho",
    "Ministerio de Salud y Protección Social",
    "Ministerio de Educación Nacional",
    "Ministerio de Minas y Energía",
    "Ministerio de Transporte",
    "Ministerio de Tecnologías de la Información y TIC",
    "Ministerio del Trabajo",
    "Ministerio de Agricultura y Desarrollo Rural",
    "Ministerio de Ambiente y Desarrollo Sostenible",
    "Ministerio de Comercio, Industria y Turismo",
    "Ministerio de Vivienda, Ciudad y Territorio",
    "Ministerio de Cultura",
    "Ministerio del Deporte",
    "Ministerio de Ciencia, Tecnología e Innovación",
    "Ministerio de Igualdad y Equidad",
    "Departamento Nacional de Planeación (DNP)",
    "Departamento Administrativo Nacional de Estadística (DANE)",
    "Departamento Administrativo de la Función Pública (DAFP)",
    "Departamento Administrativo de la Presidencia (DAPRE)",
    "Departamento para la Prosperidad Social (DPS)",
    "Dirección de Impuestos y Aduanas Nacionales (DIAN)",
    "Superintendencia Financiera de Colombia",
    "Superintendencia de Salud",
    "Superintendencia de Sociedades",
    "Superintendencia de Industria y Comercio",
    "Superintendencia de Transporte",
    "Superintendencia de Servicios Públicos Domiciliarios",
    "Superintendencia de Notariado y Registro",
    "Agencia Nacional de Infraestructura (ANI)",
    "Agencia Nacional de Minería (ANM)",
    "Agencia Nacional de Hidrocarburos (ANH)",
    "Agencia Nacional de Tierras (ANT)",
    "Agencia de Desarrollo Rural (ADR)",
    "Agencia Nacional de Seguridad Vial (ANSV)",
    "Agencia Nacional de Licencias Ambientales (ANLA)",
    "Agencia Presidencial de Cooperación Internacional (APC)",
    "Instituto Nacional de Vías (INVIAS)",
    "Instituto Colombiano de Bienestar Familiar (ICBF)",
    "Servicio Nacional de Aprendizaje (SENA)",
    "Instituto Geográfico Agustín Codazzi (IGAC)",
    "Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)",
    "Instituto Nacional de Vigilancia de Medicamentos (INVIMA)",
    "Instituto Colombiano Agropecuario (ICA)",
    "Fondo de Adaptación",
    "Unidad Nacional para la Gestión del Riesgo (UNGRD)",
    "Unidad de Planificación Rural Agropecuaria (UPRA)",
    "Unidad de Víctimas"
]


def generar_curules_no_uniforme(total_curules=50, num_partidos=5):
    """
    Genera una distribución no uniforme y aleatoria de curules entre los partidos,
    garantizando que la suma sea exactamente total_curules y que cada partido tenga al menos 1 curul.
    """
    pesos = np.random.exponential(scale=2.0, size=num_partidos) + 0.5
    proporciones = pesos / np.sum(pesos)
    
    curules = np.ones(num_partidos, dtype=int)
    restantes = total_curules - num_partidos
    
    dist_adicional = np.random.multinomial(restantes, proporciones)
    curules += dist_adicional
    return curules


def generar_pesos_entidades(num_entidades=50):
    """
    Genera pesos políticos aleatorios entre 1 y 100 para cada entidad estatal.
    """
    return np.random.randint(1, 101, size=num_entidades)


# ==============================================================================
# 2. MODELADO DEL ALGORITMO GENÉTICO (AG)
# ==============================================================================

class AlgoritmoGeneticoPoder:
    """
    Algoritmo Genético para la repartición óptima del poder político entre partidos.
    """
    def __init__(
        self,
        curules,
        pesos_entidades,
        poblacion_tam=150,
        generaciones=350,
        prob_cruce=0.85,
        prob_mutacion=0.035,
        k_torneo=3,
        elitismo_tam=4
    ):
        self.curules = np.array(curules)
        self.num_partidos = len(curules)
        self.pesos_entidades = np.array(pesos_entidades)
        self.num_entidades = len(pesos_entidades)
        
        self.poblacion_tam = poblacion_tam
        self.generaciones = generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.k_torneo = k_torneo
        self.elitismo_tam = elitismo_tam
        
        # Poder total estatal y cuota ideal de poder por partido
        self.poder_total = np.sum(self.pesos_entidades)
        self.proporcion_curules = self.curules / np.sum(self.curules)
        self.poder_objetivo = self.proporcion_curules * self.poder_total
        
        # Historial de métricas
        self.historial_mejor_fitness = []
        self.historial_mejor_error = []
        self.historial_promedio_fitness = []
        self.mejor_individuo = None
        self.mejor_fitness = -1.0
        self.mejor_error = float('inf')

    def calcular_poder_asignado(self, cromosoma):
        """
        Calcula el poder asignado a cada partido según el cromosoma.
        """
        poder_partidos = np.zeros(self.num_partidos)
        for entidad_idx, partido_idx in enumerate(cromosoma):
            poder_partidos[partido_idx] += self.pesos_entidades[entidad_idx]
        return poder_partidos

    def calcular_fitness(self, cromosoma):
        """
        Función de aptitud (Fitness):
        Inverso del error absoluto total respecto a la meta de poder proporcional.
        """
        poder_asignado = self.calcular_poder_asignado(cromosoma)
        error_total = np.sum(np.abs(poder_asignado - self.poder_objetivo))
        fitness = 1000.0 / (1.0 + error_total)
        return fitness, error_total

    def crear_individuo(self):
        """
        Genera un cromosoma aleatorio de 50 genes con valores en [0, num_partidos - 1].
        """
        return np.random.randint(0, self.num_partidos, size=self.num_entidades)

    def inicializar_poblacion(self):
        """
        Inicializa la población de individuos.
        """
        return [self.crear_individuo() for _ in range(self.poblacion_tam)]

    def seleccion_torneo(self, poblacion, fitnesses):
        """
        Selección por Torneo de tamaño k.
        """
        seleccionados_indices = np.random.choice(len(poblacion), size=self.k_torneo, replace=False)
        mejor_idx = seleccionados_indices[0]
        mejor_fit = fitnesses[mejor_idx]
        
        for idx in seleccionados_indices[1:]:
            if fitnesses[idx] > mejor_fit:
                mejor_fit = fitnesses[idx]
                mejor_idx = idx
                
        return poblacion[mejor_idx].copy()

    def cruce_uniforme(self, padre1, padre2):
        """
        Cruce uniforme entre dos padres.
        """
        if random.random() > self.prob_cruce:
            return padre1.copy(), padre2.copy()
        
        mascara = np.random.rand(self.num_entidades) < 0.5
        hijo1 = np.where(mascara, padre1, padre2)
        hijo2 = np.where(mascara, padre2, padre1)
        return hijo1, hijo2

    def mutar(self, cromosoma):
        """
        Mutación por gen: con probabilidad prob_mutacion, reasigna la entidad a otro partido.
        """
        for i in range(self.num_entidades):
            if random.random() < self.prob_mutacion:
                cromosoma[i] = random.randint(0, self.num_partidos - 1)
        return cromosoma

    def ejecutar(self):
        """
        Ejecuta el ciclo generacional del Algoritmo Genético.
        """
        poblacion = self.inicializar_poblacion()
        
        for gen in range(self.generaciones):
            # Evaluación de aptitud
            evaluaciones = [self.calcular_fitness(ind) for ind in poblacion]
            fitnesses = [ev[0] for ev in evaluaciones]
            errores = [ev[1] for ev in evaluaciones]
            
            # Identificar al mejor de la generación
            idx_mejor = int(np.argmax(fitnesses))
            if fitnesses[idx_mejor] > self.mejor_fitness:
                self.mejor_fitness = fitnesses[idx_mejor]
                self.mejor_error = errores[idx_mejor]
                self.mejor_individuo = poblacion[idx_mejor].copy()
                
            self.historial_mejor_fitness.append(self.mejor_fitness)
            self.historial_mejor_error.append(self.mejor_error)
            self.historial_promedio_fitness.append(np.mean(fitnesses))
            
            # Elitismo: ordenar población por fitness descendente
            indices_ordenados = np.argsort(fitnesses)[::-1]
            nueva_poblacion = [poblacion[i].copy() for i in indices_ordenados[:self.elitismo_tam]]
            
            # Generar el resto de la nueva población
            while len(nueva_poblacion) < self.poblacion_tam:
                padre1 = self.seleccion_torneo(poblacion, fitnesses)
                padre2 = self.seleccion_torneo(poblacion, fitnesses)
                
                hijo1, hijo2 = self.cruce_uniforme(padre1, padre2)
                
                nueva_poblacion.append(self.mutar(hijo1))
                if len(nueva_poblacion) < self.poblacion_tam:
                    nueva_poblacion.append(self.mutar(hijo2))
                    
            poblacion = nueva_poblacion
            
        return self.mejor_individuo, self.mejor_fitness, self.mejor_error

    def obtener_matriz_poder(self, cromosoma=None):
        """
        Construye la matriz binaria de asignación de poder M (5 x 50).
        M[i, j] = 1 si la entidad j se asignó al partido i, 0 en otro caso.
        """
        if cromosoma is None:
            cromosoma = self.mejor_individuo
        matriz = np.zeros((self.num_partidos, self.num_entidades), dtype=int)
        for entidad_idx, partido_idx in enumerate(cromosoma):
            matriz[partido_idx, entidad_idx] = 1
        return matriz


# ==============================================================================
# 3. GENERACIÓN DE VISUALIZACIONES Y GRÁFICOS
# ==============================================================================

def generar_graficos(ag, salida_dir="."):
    """
    Genera y guarda las gráficas de análisis del Algoritmo Genético.
    """
    os.makedirs(salida_dir, exist_ok=True)
    
    poder_asignado = ag.calcular_poder_asignado(ag.mejor_individuo)
    porc_curules = ag.proporcion_curules * 100
    porc_poder_asignado = (poder_asignado / ag.poder_total) * 100
    
    # --------------------------------------------------------------------------
    # Gráfica 1: Curva de Convergencia del AG
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(ag.historial_mejor_error, color='#c0392b', linewidth=2.2, label='Mejor Error Absoluto')
    ax1.set_title('Convergencia del Error Residual', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Error Absoluto Acumulado (Puntos)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(ag.historial_mejor_fitness, color='#27ae60', linewidth=2.2, label='Mejor Fitness')
    ax2.plot(ag.historial_promedio_fitness, color='#2980b9', linestyle='--', label='Fitness Promedio')
    ax2.set_title('Evolución del Fitness', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Fitness (Aptitud)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    ruta_conv = os.path.join(salida_dir, 'convergencia_ag.png')
    plt.savefig(ruta_conv, dpi=300)
    plt.close()
    print(f"[OK] Gráfica guardada: {ruta_conv}")

    # --------------------------------------------------------------------------
    # Gráfica 2: Comparativa de Representatividad (Curules vs Poder)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    indices = np.arange(ag.num_partidos)
    ancho = 0.35
    
    barras_curules = ax.bar(indices - ancho/2, porc_curules, ancho, label='Representación en Curules (%)', color='#2980b9', alpha=0.85)
    barras_poder = ax.bar(indices + ancho/2, porc_poder_asignado, ancho, label='Poder Estatal Asignado por AG (%)', color='#d35400', alpha=0.85)
    
    ax.set_title('Comparativa de Proporcionalidad: Curules vs Poder Asignado', fontsize=13, fontweight='bold')
    ax.set_xlabel('Partidos Políticos', fontweight='bold')
    ax.set_ylabel('Porcentaje (%)', fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"P{i+1}\n({c} curules)" for i, c in enumerate(ag.curules)], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in barras_curules:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a5276')
        
    for bar in barras_poder:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#935116')
        
    plt.tight_layout()
    ruta_comp = os.path.join(salida_dir, 'comparativa_poder.png')
    plt.savefig(ruta_comp, dpi=300)
    plt.close()
    print(f"[OK] Gráfica guardada: {ruta_comp}")

    # --------------------------------------------------------------------------
    # Gráfica 3: Matriz de Asignación de Poder (Mapa de Calor 5x50)
    # --------------------------------------------------------------------------
    matriz_poder = ag.obtener_matriz_poder()
    matriz_pesos = matriz_poder * ag.pesos_entidades.reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(16, 4.5))
    cax = ax.imshow(matriz_pesos, cmap='YlGnBu', aspect='auto')
    
    ax.set_title('Matriz de Distribución de Poder Estatal (5 Partidos x 50 Entidades)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Índice de Entidad Estatal (0 a 49)', fontweight='bold')
    ax.set_ylabel('Partidos Políticos', fontweight='bold')
    ax.set_yticks(np.arange(ag.num_partidos))
    ax.set_yticklabels([f"P{i+1}: {PARTIDOS[i][:15]}..." for i in range(ag.num_partidos)], fontsize=9)
    
    cbar = fig.colorbar(cax, orientation='horizontal', pad=0.28, shrink=0.7)
    cbar.set_label('Peso Político Asignado a la Entidad (0 = No asignado, >0 = Puntos de poder)', fontsize=9)
    
    plt.tight_layout()
    ruta_mat = os.path.join(salida_dir, 'matriz_poder.png')
    plt.savefig(ruta_mat, dpi=300)
    plt.close()
    print(f"[OK] Gráfica guardada: {ruta_mat}")


# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================

def main():
    print("=" * 80)
    print(" PUNTO 2: VERDADERA DEMOCRACIA - DISTRIBUCIÓN DE PODER CON ALGORITMOS GENÉTICOS")
    print("=" * 80)
    
    # 1. Generar Curules no uniformes
    curules = generar_curules_no_uniforme(TOTAL_CURULES, NUM_PARTIDOS)
    print("\n--- DISTRIBUCIÓN DE CURULES EN EL CONGRESO (TOTAL: 50) ---")
    for i, (partido, c) in enumerate(zip(PARTIDOS, curules)):
        porc = (c / TOTAL_CURULES) * 100
        print(f"  [Partido {i+1}] {partido:<38}: {c:>2} curules ({porc:>5.1f}%)")
        
    # 2. Generar Pesos de Entidades
    pesos_entidades = generar_pesos_entidades(TOTAL_ENTIDADES)
    poder_total = np.sum(pesos_entidades)
    print(f"\n--- 50 ENTIDADES ESTATALES GENERADAS ---")
    print(f"  Poder Político Total del Estado: {poder_total} puntos")
    print(f"  Peso promedio por entidad: {np.mean(pesos_entidades):.2f} pts | Mín: {np.min(pesos_entidades)} | Máx: {np.max(pesos_entidades)}")
    
    # 3. Configurar y Ejecutar AG
    print("\n--- EJECUTANDO ALGORITMO GENÉTICO ---")
    ag = AlgoritmoGeneticoPoder(
        curules=curules,
        pesos_entidades=pesos_entidades,
        poblacion_tam=150,
        generaciones=350,
        prob_cruce=0.85,
        prob_mutacion=0.035,
        k_torneo=3,
        elitismo_tam=4
    )
    
    mejor_cromosoma, mejor_fitness, mejor_error = ag.ejecutar()
    
    print(f"  Generaciones completadas: {ag.generaciones}")
    print(f"  Mejor Fitness alcanzado : {mejor_fitness:.4f}")
    print(f"  Error Absoluto Acumulado: {mejor_error:.2f} puntos de poder (sobre {poder_total} pts totales)")
    
    # 4. Tabla de Resultados Comparativos
    poder_asignado = ag.calcular_poder_asignado(mejor_cromosoma)
    matriz_poder = ag.obtener_matriz_poder()
    entidades_por_partido = np.sum(matriz_poder, axis=1)
    
    print("\n" + "=" * 95)
    print(f"{'Partido':<38} | {'Curules':<7} | {'% Curules':<9} | {'Poder Obj':<9} | {'Poder Asig':<10} | {'% Poder':<7} | {'Error':<5} | {'# Ent.'}")
    print("=" * 95)
    
    for i in range(NUM_PARTIDOS):
        porc_c = (curules[i] / TOTAL_CURULES) * 100
        p_obj = ag.poder_objetivo[i]
        p_asig = poder_asignado[i]
        porc_p = (p_asig / poder_total) * 100
        err = abs(p_asig - p_obj)
        num_ent = entidades_por_partido[i]
        
        print(f"{PARTIDOS[i]:<38} | {curules[i]:<7} | {porc_c:>7.1f}% | {p_obj:>9.1f} | {p_asig:>10.1f} | {porc_p:>6.1f}% | {err:>5.1f} | {num_ent:>5}")
        
    print("=" * 95)
    
    # 5. Generar y Guardar Gráficos
    salida_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n--- GENERANDO GRÁFICAS DE RESULTADOS ---")
    generar_graficos(ag, salida_dir)
    
    print("\n[ÉXITO] Ejecución completada satisfactoriamente.")


if __name__ == "__main__":
    main()
