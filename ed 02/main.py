import random
from dataclasses import dataclass
from typing import List

@dataclass
class Aluno:
    nome: str
    matricula: str
    notas: List[float]

    def get_media(self) -> float:
        if not self.notas:
            return 0.0
        return sum(self.notas) / len(self.notas)

# Função para gerar notas aleatórias entre 0 e 10
def gerar_notas_aleatorias():
    return [random.uniform(0, 10) for _ in range(4)]

# Criar uma lista de 10 alunos com notas aleatórias
alunos = [
    Aluno(nome="baAluno1", matricula="12345", notas=gerar_notas_aleatorias()),
    Aluno(nome="wAluno2", matricula="54321", notas=gerar_notas_aleatorias()),
    Aluno(nome="qAluno3", matricula="98765", notas=gerar_notas_aleatorias()),
    Aluno(nome="dAluno4", matricula="67890", notas=gerar_notas_aleatorias()),
    Aluno(nome="zAluno5", matricula="24680", notas=gerar_notas_aleatorias()),
    Aluno(nome="sAluno6", matricula="13579", notas=gerar_notas_aleatorias()),
    Aluno(nome="Aluno7", matricula="11223", notas=gerar_notas_aleatorias()),
    Aluno(nome="Aluno8", matricula="998877", notas=gerar_notas_aleatorias()),
    Aluno(nome="cAluno9", matricula="55555", notas=gerar_notas_aleatorias()),
    Aluno(nome="eAluno10", matricula="11111", notas=gerar_notas_aleatorias()),
]

# Calcular as médias dos alunos
for aluno in alunos:
    aluno.media = aluno.get_media()

# Função para obter a média de um aluno (usada como chave de classificação)
def obter_media_aluno(aluno):
    return aluno.media

# Classificar a lista de alunos pelo nome
alunos_ordenados_nome = sorted(alunos, key=lambda x: x.nome)

# Classificar a lista de alunos pela média (nota) decrescente
alunos_ordenados_nota = sorted(alunos, key=obter_media_aluno, reverse=True)

# Exemplo de uso: imprimir a média das notas dos alunos em ordem alfabética
print("Alunos em ordem alfabética:")
for aluno in alunos_ordenados_nome:
    print(f"Nome: {aluno.nome}, Matrícula: {aluno.matricula}, Média: {aluno.media:.2f}")

print("\nAlunos em ordem de nota decrescente:")
for aluno in alunos_ordenados_nota:
    print(f"Nome: {aluno.nome}, Matrícula: {aluno.matricula}, Média: {aluno.media:.2f}")
# Função para imprimir os 2 alunos com maior média
def imprimir_maiores_medias(alunos):
    print("Os 2 alunos com as maiores médias:")
    for aluno in alunos[:2]:
        print(f"Nome: {aluno.nome}, Matrícula: {aluno.matricula}, Média: {aluno.media:.2f}")

# Função para imprimir os 2 alunos com menor média
def imprimir_menores_medias(alunos):
    print("\nOs 2 alunos com as menores médias:")
    for aluno in alunos[-2:]:
        print(f"Nome: {aluno.nome}, Matrícula: {aluno.matricula}, Média: {aluno.media:.2f}")

# Exemplo de uso: imprimir os 2 alunos com maiores e menores médias
imprimir_maiores_medias(alunos_ordenados_nota)
imprimir_menores_medias(alunos_ordenados_nota)

# Função para calcular a média da turma
def calcular_media_turma(alunos):
    if not alunos:
        return 0.0
    total_media = sum(aluno.media for aluno in alunos)
    return total_media / len(alunos)

# Exemplo de uso: calcular a média da turma e imprimir
media_turma = calcular_media_turma(alunos_ordenados_nota)
print(f"\nMédia da turma: {media_turma:.2f}")

#função para imprimir alunos com media baixa

print("Alunos com medias baixas")
for aluno in alunos:
    if aluno.media < 6.00:
        print(f"Nome: {aluno.nome}, Matrícula: {aluno.matricula}, Média: {aluno.media:.2f}")








