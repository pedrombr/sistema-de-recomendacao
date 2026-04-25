import pandas as pd
from importdataset import carregar_dataset

dataset_notas = carregar_dataset()

"""
• Usuários 𝑢 ∈ 1, … , 𝑈
• Itens 𝑖 ∈ 1, … , 𝐼
• Conjunto de Treinamento T com preferências observadas 𝑟
𝑢𝑖
 e com valores reais para
algumas combinações de usuário-item  𝑢, 𝑖
• 𝑟
𝑢𝑖
= e. g. compra, avaliação, contagem de clicks, …
• 𝑅  é a quantidade de avaliações
• 𝑈  é a quantidade de usuários no sistema
• 𝐼  é a quantidade de itens no sistema
• 𝜇 é a média global
• 𝑈
𝑖
= Usuários que avaliaram o item 𝑖
• 𝐼
𝑢
= Itens que foram avaliados pelo usuário u
• 𝑈
𝑖𝑗
= 𝑈
𝑖
∩ 𝑈
𝑗
=Usuários que avaliaram os itens 𝑖 e 𝑗
• 𝐼
𝑢𝑣
= 𝐼
𝑢
∩ 𝐼
𝑣
=Itens que foram avaliados pelos usuários 𝑢 e 𝑣
• 𝑁 𝑢 =Conjuntos de vizinhos de 𝑢

"""

def knn_user_based(dataset):
