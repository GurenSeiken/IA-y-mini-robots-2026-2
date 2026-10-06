#!pip install deap matplotlib numpy

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
import operator

# Tamaño de la sala e ingenieros
GRID_SIZE = 10
NUM_ENGINEERS = 15

class Simulator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        # Colocar ingenieros (representados por 1)
        engineers_placed = 0
        while engineers_placed < NUM_ENGINEERS:
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if self.grid[y, x] == 0:
                self.grid[y, x] = 1
                engineers_placed += 1
        self.robot_pos = [0, 0]
        self.robot_dir = 0 # 0: derecha, 1: abajo, 2: izquierda, 3: arriba
        self.cookies_delivered = 0
        self.steps = 0
        self.max_steps = 200
        self.path = [(0,0)]

    def get_forward_pos(self):
        x, y = self.robot_pos
        if self.robot_dir == 0: x += 1
        elif self.robot_dir == 1: y += 1
        elif self.robot_dir == 2: x -= 1
        elif self.robot_dir == 3: y -= 1
        return [x, y]

    def is_engineer_ahead(self):
        fx, fy = self.get_forward_pos()
        if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
            return self.grid[fy, fx] == 1
        return False

    def move_forward(self):
        if self.steps >= self.max_steps: return
        self.steps += 1
        fx, fy = self.get_forward_pos()
        if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
            self.robot_pos = [fx, fy]
            self.path.append(tuple(self.robot_pos))
            if self.grid[fy, fx] == 1:
                self.grid[fy, fx] = 0 # Entregó galleta
                self.cookies_delivered += 1

    def turn_left(self):
        if self.steps >= self.max_steps: return
        self.steps += 1
        self.robot_dir = (self.robot_dir - 1) % 4

    def turn_right(self):
        if self.steps >= self.max_steps: return
        self.steps += 1
        self.robot_dir = (self.robot_dir + 1) % 4

# Funciones de envoltorio (wrapper) para el control de flujo en el árbol sintáctico
import functools
def progn(*args):
    for arg in args:
        arg() # Ejecuta las funciones que recibe

def prog2(out1, out2):
    return functools.partial(progn, out1, out2)

def prog3(out1, out2, out3):
    return functools.partial(progn, out1, out2, out3)

# La implementación de PG requiere que las primitivas manipulen el simulador.
# Para evitar problemas de ámbito de DEAP, usaremos funciones con estado global.
import copy

current_sim = None

def m_move():
    global current_sim
    if current_sim: current_sim.move_forward()

def m_left():
    global current_sim
    if current_sim: current_sim.turn_left()

def m_right():
    global current_sim
    if current_sim: current_sim.turn_right()

def m_sense():
    global current_sim
    return current_sim.is_engineer_ahead() if current_sim else False

def m_if(out1, out2):
    def wrapper():
        if m_sense(): out1()
        else: out2()
    return wrapper

# DEAP setup
pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(prog2, 2, name="prog2")
pset.addPrimitive(prog3, 3, name="prog3")
pset.addPrimitive(m_if, 2, name="if_engineer_ahead")
pset.addTerminal(m_move, name="Avanzar")
pset.addTerminal(m_left, name="GirarIzq")
pset.addTerminal(m_right, name="GirarDer")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalRobotGlobal(individual):
    global current_sim
    routine = gp.compile(individual, pset)
    current_sim = Simulator()
    
    while current_sim.steps < current_sim.max_steps and current_sim.cookies_delivered < NUM_ENGINEERS:
        routine()
        
    return current_sim.cookies_delivered,

toolbox.register("evaluate", evalRobotGlobal)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

from deap import algorithms

pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

print("Iniciando evolución...")
pop, logbook = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=stats, halloffame=hof, verbose=True)
print("¡Evolución terminada!")

best_ind = hof[0]
print("Mejor programa genético:\n", best_ind)

# Simulamos con el mejor individuo
current_sim = Simulator()
routine = gp.compile(best_ind, pset)
while current_sim.steps < current_sim.max_steps and current_sim.cookies_delivered < NUM_ENGINEERS:
    routine()

print(f"Galletas entregadas: {current_sim.cookies_delivered}/{NUM_ENGINEERS} en {current_sim.steps} pasos")

# Graficar ruta
path = np.array(current_sim.path)
plt.figure(figsize=(6,6))
plt.plot(path[:,0], path[:,1], marker='.', color='blue', alpha=0.5, label='Ruta Robot')
plt.scatter(path[0,0], path[0,1], color='green', s=100, label='Inicio')
plt.xlim(-1, GRID_SIZE)
plt.ylim(-1, GRID_SIZE)
plt.gca().invert_yaxis() # Para que (0,0) esté arriba a la izquierda como en matrices
plt.title("Recorrido del Robot Repartidor de Galletas")
plt.legend()
plt.grid(True)
plt.show()
