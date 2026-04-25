import pandas as pd
from importdataset import carregar_dataset_rating

dataset_notas = carregar_dataset_rating()

filmes_avaliados = []
qnt_filmes = input("Quantos filmes deseja avaliar?\n")

for i in range(int(qnt_filmes)):
  filme = input(f"Qual o {i+1}º filme deseja avaliar?\n")
  nota_filme = float(input(f"Qual a nota de {filme} (0-10)?\n"))
  filmes_avaliados.append({
      'filme': filme,
      'nota_filme': nota_filme
      })

for filme_info in filmes_avaliados:
  print(f"Filme: {filme_info['filme']}, Nota: {filme_info['nota_filme']}")

def knn_user_based(dataset):
    vizinhos_avaliaram_item =
