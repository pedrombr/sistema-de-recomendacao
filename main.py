import numpy as np
from algoritmos.knn import knn_user_based, knn_item_based, prever_nota_user_based, prever_nota_item_based
from algoritmos import svd_byfunk
from importdataset import carregar_dataset_movies, carregar_dataset_user, carregar_dataset_rating
from divisaodata import dividir_treino_val_teste
from metricas import rmse
from hiperparametros import otimizar_knn, otimizar_svd

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

    # Cálculo da média para o Item-Based
    medias_treino = matriz_treino.replace(0, np.nan).mean(axis=1)

    # User-Based
    recomendacoes = knn_user_based(matriz_treino=matriz_treino, user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)
    print("User based")
    if isinstance(recomendacoes, str): # Caso retorne mensagem de erro
        print(recomendacoes)
    else:
        print("\n Top Filmes Recomendados:")
        for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes, 1):
            print(f"{posicao}º Lugar | Filme ID: {filme_id}")

    print("---------------------------------------------------------------------")
    print("Item based")

    # Item-Based
    recomendacoes_item = knn_item_based(matriz_treino=matriz_treino, medias_treino=medias_treino, user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)

    if isinstance(recomendacoes_item, str): # Caso retorne mensagem de erro
        print(recomendacoes_item)
    else:
        print("\n Top Filmes Recomendados:")
        for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes_item, 1):
            print(f"{posicao}º Lugar | Filme ID: {filme_id}")

    # Validação (hiperparametrização)
    # lista de k para testar no KNN
    valores_k_para_testar = [5, 10, 15, 20]
    melhor_k_user, melhor_k_item = otimizar_knn(matriz_treino, medias_treino, validacao, valores_k_para_testar)

    # lista de regularizações (reg) para testar no SVD
    valores_reg_para_testar = [0.01, 0.02, 0.05, 0.1]
    melhor_reg_svd = otimizar_svd(treino, validacao, valores_reg_para_testar, fator_fixo=10)

    #print(f"\nComeçando o SVD\n")


    #SVD
    #model = svd_byfunk.SVD()
    #model.train(treino)

#| Nota Prevista: {nota_prevista:.2f}
if __name__ == "__main__":
    main()
