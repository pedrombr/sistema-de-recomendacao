import numpy as np
from algoritmos.knn import knn_user_based, knn_item_based, prever_nota_user_based, prever_nota_item_based
from importdataset import carregar_dataset_movies, carregar_dataset_user, carregar_dataset_rating
from divisaodata import dividir_treino_val_teste

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

    recomendacoes_item = knn_item_based(matriz_treino=matriz_treino, medias_treino=medias_treino, user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)

    if isinstance(recomendacoes_item, str): # Caso retorne mensagem de erro
        print(recomendacoes_item)
    else:
        print("\n Top Filmes Recomendados:")
        for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes_item, 1):
            print(f"{posicao}º Lugar | Filme ID: {filme_id}")

    # previsao e erro 
    print("\n=====================================================================")
    print("Iniciando a Hiperparametrização (Validação)")
    
    # lista de k para testar
    valores_k = [5, 10, 15, 20]
    
    # dicionários para guardar os resultados 
    historico_mae_user = {}
    historico_mae_item = {}

    # loop da Hiperparametrização
    for k_atual in valores_k:
        print(f"\n--- Testando algoritmo para k = {k_atual} vizinhos ---")
        
        erros_user_based = []
        erros_item_based = []

        # O [:100] continua aqui por enquanto só para testarmos rápido!
        for index, linha in validacao[:100].iterrows():
            usuario_alvo = int(linha['UserID'])
            filme_alvo = int(linha['MovieID'])
            nota_real = linha['Rating']

            prev_user = prever_nota_user_based(matriz_treino, usuario_alvo, filme_alvo, k_vizinhos=k_atual)
            if prev_user > 0: 
                erros_user_based.append(abs(nota_real - prev_user))

            prev_item = prever_nota_item_based(matriz_treino, medias_treino, usuario_alvo, filme_alvo, k_vizinhos=k_atual)
            if prev_item > 0:
                erros_item_based.append(abs(nota_real - prev_item))

        # calculando o MAE para o K atual
        mae_user = sum(erros_user_based) / len(erros_user_based) if erros_user_based else 0
        mae_item = sum(erros_item_based) / len(erros_item_based) if erros_item_based else 0
        
        # guardar a nota final do K no histórico
        historico_mae_user[k_atual] = mae_user
        historico_mae_item[k_atual] = mae_item

        print(f"Erro Médio (MAE) User-Based: {mae_user:.4f} estrelas")
        print(f"Erro Médio (MAE) Item-Based: {mae_item:.4f} estrelas")

#| Nota Prevista: {nota_prevista:.2f}
if __name__ == "__main__":
    main()

# p999999999999
