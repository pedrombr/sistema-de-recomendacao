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

    usuario_teste = 10

    # Transformando a lista em Matriz Usuário x Filme
    matriz_treino = treino.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    matriz_teste = teste.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    # Cálculo da média para o Item-Based
    medias_treino = matriz_treino.replace(0, np.nan).mean(axis=1)

    print("User based")
    inicio_user = time.time()
    modelo_knn_user = KNNUserBased(k_vizinhos=15, metrica='pearson')
    modelo_knn_user.fit(matriz_treino)

    recomendacoes_user = modelo_knn_user.knn_user_based(matriz_treino,user_id=1)

    for posicao, (filme_id, nota) in enumerate(recomendacoes_user, start=1):
        print(f"{posicao}º filme -> ID: {filme_id} | Nota prevista: {nota:.2f}")

    fim_user = time.time()
    tempo_knn_user = fim_user - inicio_user
    print(f'Tempo: {tempo_knn_user:.2f}s')

    erro_user = modelo_knn_user.avaliar_rmse_user_based(matriz_treino,matriz_teste)
    print(f'RMSE: {erro_user:.4f}')

    print("------------------------------")
    #print("Item based")
    #inicio_item = time.time()
    #modelo_knn_item = KNNItemBased(k_vizinhos=5,metrica='cosseno_ajustado')

    #modelo_knn_item.fit(matriz_treino,medias_treino)

    #recomendacoes_item = modelo_knn_item.knn_item_based(matriz_treino,medias_treino,user_id=1)
    #for posicao, (filme_id, nota) in enumerate(recomendacoes_item,start=1):
    #    print(f'{posicao}º filme -> 'f'ID: {filme_id} | 'f'Nota prevista: {nota:.2f}')

    #fim_item = time.time()
    #tempo_knn_item = fim_item - inicio_item
    #print(f'Tempo: {tempo_knn_item:.2f}s')

    #erro_item = modelo_knn_item.avaliar_rmse_item_based(matriz_treino,matriz_teste,medias_treino)
    #print(f'RMSE: {erro_item:.4f}')

    print("------------------------------")
    #print("\nComeçando o SVD\n")
    #SVD
    ##inicio_svd = time.time()
    ##model = svd_byfunk.SVD()
    ##model.train(treino)
    ##fim_svd = time.time()
    ##tempo_svd = fim_svd - inicio_svd
    ##print(f'Tempo: {tempo_svd:.2f}s')

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

if __name__ == "__main__":
    main()
