from dataclasses import dataclass
from typing import Callable
from random import randint, sample, random
from pprint import pprint

@dataclass
class Cromossomo:
    """Representa um cromossomo no algoritmo genético."""
    dados: list[int]  # Representa os genes do cromossomo
    fitness: float = 0  # Aptidão deste cromossomo

@dataclass
class Config:
    """Configurações para a execução do algoritmo genético."""
    tam_cromossomo: int  # Tamanho dos cromossomos
    tam_populacao: int  # Tamanho da população
    fitness: Callable[[Cromossomo], float]  # Função de avaliação

    # Função para selecionar os pais para cruzamento
    selecionar_pais: Callable[[list[Cromossomo]], list[tuple[Cromossomo, Cromossomo]]]

    # Função para aplicar cruzamento entre pais
    aplicar_cruzamento: Callable[[list[tuple[Cromossomo, Cromossomo]]], list[Cromossomo]]

    # Função para aplicar mutação nos cromossomos
    aplicar_mutacao: Callable[[list[Cromossomo], float], None]
    taxa_mutacao: float  # Taxa de mutação

    # Função para selecionar os sobreviventes da próxima geração
    selecionar_sobreviventes: Callable[[list[Cromossomo]], list[Cromossomo]]

def inicializar_populacao(tam_populacao: int, tam_cromossomo: int) -> list[Cromossomo]:
    """Inicializa uma população de cromossomos com genes aleatórios."""
    return [Cromossomo([randint(0, 1) for _ in range(tam_cromossomo)]) for _ in range(tam_populacao)]

def torneio(populacao: list[Cromossomo], tam_torneio: int = 3) -> list[tuple[Cromossomo, Cromossomo]]:
    """Seleciona pais para cruzamento usando o método do torneio."""
    casais = []
    for _ in range(len(populacao) // 2):
        pai1 = max(sample(populacao, tam_torneio * 2), key=lambda x: x.fitness)
        populacao.remove(pai1)

        pai2 = max(sample(populacao, tam_torneio), key=lambda x: x.fitness)
        casais.append((pai1, pai2))
    return casais

def crossover_1_corte(casais: list[tuple[Cromossomo, Cromossomo]]) -> list[Cromossomo]:
    """Aplica o cruzamento de um ponto entre os pais selecionados."""
    filhos = []
    for pai1, pai2 in casais:
        corte = randint(1, len(pai1.dados) - 1)
        filhos.extend([
            Cromossomo(pai1.dados[:corte] + pai2.dados[corte:]),
            Cromossomo(pai2.dados[:corte] + pai1.dados[corte:])
        ])
    return filhos

def mutacao(cromossomos: list[Cromossomo], taxa_mutacao: float):
    """Aplica mutação em um cromossomo com base em uma taxa de mutação."""
    for cromossomo in cromossomos:
        for i, gene in enumerate(cromossomo.dados):
            if random() < taxa_mutacao:
                cromossomo.dados[i] = 1 - gene  # Inverte o gene (0 vira 1 e 1 vira 0)

def elitismo(populacao: list[Cromossomo]) -> list[Cromossomo]:
    """Seleciona a metade mais apta da população."""
    return sorted(populacao, key=lambda x: x.fitness, reverse=True)[:len(populacao)//2]

def calcular_fitness(cromossomo: Cromossomo) -> float:
    """Função de avaliação para o Problema dos Cartões."""
    pilha1 = []  # Inicializa a primeira pilha vazia
    pilha2 = list(range(1, 11))  # Inicializa a segunda pilha com todos os cartões

    for i, gene in enumerate(cromossomo.dados):
        if gene == 1:
            pilha1.append(i + 1)  # Adiciona o cartão à primeira pilha
            pilha2.remove(i + 1)  # Remove o cartão da segunda pilha

    # Calcula a soma dos cartões na primeira pilha
    soma_pilha1 = sum(pilha1)

    # Calcula o produto dos cartões na segunda pilha
    produto_pilha2 = 1
    for carta in pilha2:
        produto_pilha2 *= carta

    # Calcula as diferenças em relação aos objetivos
    diff_soma = abs(soma_pilha1 - 36)
    diff_produto = abs(produto_pilha2 - 360)

    # Quanto mais próximo de 0, melhor
    return diff_soma + diff_produto

def algoritmo_genetico(config: Config) -> list[int]:
    """Executa o algoritmo genético com base nas configurações fornecidas."""
    t = 0
    P = inicializar_populacao(config.tam_populacao, config.tam_cromossomo)

    while True:  # Critério de parada pode ser adicionado aqui
        print("\n> Avaliação da população")
        for c in P:
            c.fitness = config.fitness(c)
            print(c)

        print("\n> Seleção dos pais")
        casais = config.selecionar_pais(P)
        for c in casais:
            print(c)

        print("\n> Crossover e mutação")
        F = config.aplicar_cruzamento(casais)
        config.aplicar_mutacao(F, config.taxa_mutacao)
        for c in F:
            print(c)

        print("\n> Avaliação da nova população")
        for c in F:
            c.fitness = config.fitness(c)
            print(c)

        print("\n> Selecionar sobreviventes")
        P = config.selecionar_sobreviventes(P + F)
        for c in P:
            print(c)

        t += 1
        break  # Este é um placeholder; em uma implementação real, você deve definir um critério de parada adequado.


config = Config(
    tam_cromossomo=10,  # 10 cartões numerados de 1 a 10
    tam_populacao=50,  # Tamanho da população
    fitness=calcular_fitness,  # Usar a função de fitness para o Problema dos Cartões
    taxa_mutacao=0.05,
    selecionar_pais=torneio,
    aplicar_cruzamento=crossover_1_corte,
    aplicar_mutacao=mutacao,
    selecionar_sobreviventes=elitismo,
)
solucao = algoritmo_genetico(config)
