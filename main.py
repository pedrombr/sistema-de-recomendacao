import time
import numpy as np
from algoritmos.knn_item import KNNItemBased
from algoritmos.knn_user import KNNUserBased
from algoritmos import svd_byfunk
from importdataset import carregar_dataset_movies, carregar_dataset_user, carregar_dataset_rating
from divisaodata import dividir_treino_val_teste
from metricas import rmse

def main():
    # carregar os dados
    df_ratings = carregar_dataset_rating()

    # dividir os dados
    treino, validacao, teste = dividir_treino_val_teste(df_ratings, 0.8, 0.1)

    print(f"Tamanho do Treino: {len(treino)} linhas")
    print(f"Tamanho da Validação: {len(validacao)} linhas")
    print(f"Tamanho do Teste: {len(teste)} linhas")

    # Transformando a lista em Matriz Usuário x Filme
    matriz_treino = treino.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    matriz_teste = teste.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    matriz_validacao = validacao.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

    # Cálculo da média para o Item-Based
    medias_treino = matriz_treino.replace(0, np.nan).mean(axis=1)
    #[10, 15, 20, 25, 27, 30, 35, 40]

    """inicio_user = time.time()
    modelo_knn_user = KNNUserBased(k_vizinhos=30, metrica='pearson_unha')
    print("--- Configurações do KNN user based ---")
    print(f"K vizinhos (k): {modelo_knn_user.k_vizinhos}")
    print(f"Métrica usada: {modelo_knn_user.metrica}")
    print("-----------------------------------")
    modelo_knn_user.fit(matriz_treino)

    recomendacoes_user = modelo_knn_user.knn_user_based(matriz_treino,user_id=1)

    for posicao, (filme_id, nota) in enumerate(recomendacoes_user, start=1):
        print(f"{posicao}º filme -> ID: {filme_id} | Nota prevista: {nota:.2f}")

    fim_user = time.time()
    tempo_knn_user = fim_user - inicio_user
    print(f'Tempo: {tempo_knn_user:.2f}s')

    erro_user = modelo_knn_user.avaliar_rmse_user_based(matriz_treino, matriz_treino)
    print(f'RMSE Treino: {erro_user:.4f}')
    print("------------------------------")
    rmse_validacao = modelo_knn_user.avaliar_rmse_user_based(matriz_treino, matriz_validacao)
    print(f'RMSE Validação: {rmse_validacao:.4f}')
    print("------------------------------")
    rmse_teste = modelo_knn_user.avaliar_rmse_user_based(matriz_treino, matriz_teste)
    print(f'RMSE Teste: {rmse_teste:.4f}')"""

    print("------------------------------")
    print("Item based")
    inicio_item = time.time()
    modelo_knn_item = KNNItemBased(k_vizinhos=100, metrica='cosseno_ajustado')
    print("--- Configurações do KNN item based ---")
    print(f"K vizinhos (k): {modelo_knn_item.k_vizinhos}")
    print(f"Métrica usada: {modelo_knn_item.metrica}")
    print("-----------------------------------")

    modelo_knn_item.fit(matriz_treino)

    recomendacoes_item = modelo_knn_item.item_based(matriz_treino, user_id=1)
    for posicao, (filme_id, nota) in enumerate(recomendacoes_item,start=1):
        print(f'{posicao}º filme -> 'f'ID: {filme_id} | 'f'Nota prevista: {nota:.2f}')

    fim_item = time.time()
    tempo_knn_item = fim_item - inicio_item
    print(f'Tempo: {tempo_knn_item:.2f}s')

    erro_item_treino = (modelo_knn_item.avaliar_rmse_item_based(matriz_treino, matriz_treino))
    print(f'RMSE teste: {erro_item_treino:.4f}')
    print("------------------------------")
    erro_item_validacao = (modelo_knn_item.avaliar_rmse_item_based(matriz_treino, matriz_validacao))
    print(f'RMSE Validação: {erro_item_validacao:.4f}')
    print("------------------------------")
    erro_item_teste = (modelo_knn_item.avaliar_rmse_item_based(matriz_treino, matriz_teste))

    print(f'RMSE Teste: {erro_item_teste:.4f}')

    """print("------------------------------")
    inicio_svd = time.time()

    modelo_svd = svd_byfunk.SVD(reg=0.02, k_factors=200, epochs=20, learning_rate=0.005)
    print("--- Configurações do Modelo SVD ---")
    print(f"Fatores Latentes (k): {modelo_svd.k_factors}")
    print(f"Épocas de Treino: {modelo_svd.n_epochs}")
    print(f"Taxa de Aprendizado: {modelo_svd.learning_rate}")
    print(f"Termo de Regularização: {modelo_svd.reg}")
    print("-----------------------------------")

    modelo_svd.train(treino)

    fim_svd = time.time()
    tempo_svd = fim_svd - inicio_svd
    print(f'Tempo de execução: {tempo_svd:.2f}s')

    rmse_treino_svd = modelo_svd.avaliar_rmse_svd(treino)
    print(f'RMSE Treino: {rmse_treino_svd:.4f}')
    print("------------------------------")

    rmse_validacao_svd = modelo_svd.avaliar_rmse_svd(validacao)
    print(f'RMSE Validação: {rmse_validacao_svd:.4f}')
    print("------------------------------")

    rmse_teste_svd = modelo_svd.avaliar_rmse_svd(teste)
    print(f'RMSE Teste: {rmse_teste_svd:.4f}')
    print("------------------------------")"""


if __name__ == "__main__":
    main()


# Nota Prevista: {nota_prevista:.2f}

# pré refatoração
# User-Based
# recomendacoes = knn_user_based(matriz_treino=matriz_treino, user_id=usuario_teste, k_vizinhos=20, n_recomendacoes=5, normalizar=False)
# print("User based")
# print("\n Top Filmes Recomendados:")
# for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes, 1):
#   print(f"{posicao}º Lugar | Filme ID: {filme_id}")
# Item-Based
# recomendacoes_item = knn_item_based(matriz_treino=matriz_treino, medias_treino=medias_treino, user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)

# print("\n Top Filmes Recomendados:")
# for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes_item, 1):
#    print(f"{posicao}º Lugar | Filme ID: {filme_id}")
