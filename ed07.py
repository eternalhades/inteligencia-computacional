from __future__ import annotations
from random import uniform, random
from typing import List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import math
import matplotlib.pyplot as plt

class TipoOtimizacao(Enum):
    MIN = 0
    MAX = 1

@dataclass
class Particula:
    n: int
    x: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    fitness: float = 0
    pbest: 'Particula' = None
    gbest: 'Particula' = None
    dominio: List[Tuple[int, int]] = field(default_factory=list)
    tipo_otimizacao: TipoOtimizacao = TipoOtimizacao.MAX

    def __post_init__(self):
        if not self.x:
            self.x = [uniform(self.dominio[i][0], self.dominio[i][1]) for i in range(self.n)]
            self.v = [random() for _ in range(self.n)]
        self.pbest = self

        if self.tipo_otimizacao == TipoOtimizacao.MIN:
            self.fitness = float('inf')
        else:
            self.fitness = float('-inf')

    def eh_melhor(self, outra: 'Particula') -> bool:
        if self.tipo_otimizacao == TipoOtimizacao.MIN:
            return self.fitness < outra.fitness
        return self.fitness > outra.fitness

def funcao_fitness(p: Particula) -> float:
    x, y = p.x[0], p.x[1]
    return -(x**2 / 2) - (x**2 / 2) + x - 2*y + math.cos(x*y)

def pso_maximizar(w: float,
                  c1: float,
                  c2: float,
                  tam_populacao: int,
                  max_iteracoes: int,
                  fitness: Callable[[Particula], float],
                  dominio: List[Tuple[int, int]] = None) -> Particula:
    P = [Particula(n=len(dominio), dominio=dominio, tipo_otimizacao=TipoOtimizacao.MAX) for _ in range(tam_populacao)]
    gbest = deepcopy(P[0])

    for k in range(max_iteracoes):
        for p in P:
            p.fitness = fitness(p)
            if p.eh_melhor(p.pbest):
                p.pbest = deepcopy(p)

        for p in P:
            if p.eh_melhor(gbest):
                gbest = deepcopy(p)

        for p in P:
            p.gbest = gbest

        for p in P:
            for i in range(p.n):
                p.v[i] = w * p.v[i] + c1 * random() * (p.pbest.x[i] - p.x[i]) + c2 * random() * (p.gbest.x[i] - p.x[i])
                p.x[i] += p.v[i]

    return gbest

# Configuração do PSO
w = 0.5
c1 = 1.5
c2 = 1.5
tam_populacao = 30
max_iteracoes = 50

# Configuração do domínio
dominio = [(-5, 5), (-5, 5)]

# Execução do PSO
melhor_solucao = pso_maximizar(w, c1, c2, tam_populacao, max_iteracoes, funcao_fitness, dominio)

# Resultados
print(f"Melhor solução encontrada: {melhor_solucao.x}")
print(f"Valor máximo da função: {melhor_solucao.fitness}")

# Visualização da função e da solução
valores_x = valores_y = list(range(-5, 6))
valores_z = [[x, y, -funcao_fitness(Particula(x=[x, y], n=len(dominio)))] for x in valores_x for y in valores_y]
X, Y, Z = zip(*valores_z)

figura = plt.figure()
eixo = figura.add_subplot(projection='3d')
eixo.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.1, alpha=0.7)

eixo.scatter(melhor_solucao.x[0], melhor_solucao.x[1], -melhor_solucao.fitness, color='red', s=100, label='Melhor Solução')
eixo.set_xlabel('X')
eixo.set_ylabel('Y')
eixo.set_zlabel('f(X, Y)')
plt.legend()
plt.show()
