import numpy as np
from metricas import rmse
from algoritmos.knn import prever_nota_user_based, prever_nota_item_based
from algoritmos.svd_byfunk import SVD

def otimizar_knn(matriz_treino, medias_treino, base_validacao, valores_k):
    print("\n=====================================================================")
    print("Iniciando a Hiperparametrização do KNN (Validação)")
    
    historico_rmse_user = {}
    historico_rmse_item = {}

    for k_atual in valores_k:
        print(f"\n--- Testando algoritmo para k = {k_atual} vizinhos ---")

        lista_prev_user = []
        lista_real_user = []
        
        lista_prev_item = []
        lista_real_item = []

        for index, linha in base_validacao.iterrows():
            usuario_alvo = int(linha['UserID'])
            filme_alvo = int(linha['MovieID'])
            nota_real = linha['Rating']

            # User-Based
            prev_user = prever_nota_user_based(matriz_treino, usuario_alvo, filme_alvo, k_vizinhos=k_atual)
            if prev_user > 0:
                lista_prev_user.append(prev_user)
                lista_real_user.append(nota_real)

            # Item-Based
            prev_item = prever_nota_item_based(matriz_treino, medias_treino, usuario_alvo, filme_alvo, k_vizinhos=k_atual)
            if prev_item > 0:
                lista_prev_item.append(prev_item)
                lista_real_item.append(nota_real)

        # Cálculo do Erro
        rmse_user = rmse(np.array(lista_prev_user), np.array(lista_real_user))
        rmse_item = rmse(np.array(lista_prev_item), np.array(lista_real_item))

        historico_rmse_user[k_atual] = rmse_user
        historico_rmse_item[k_atual] = rmse_item

        print(f"Erro Médio (RMSE) User-Based: {rmse_user:.4f}")
        print(f"Erro Médio (RMSE) Item-Based: {rmse_item:.4f}")

    melhor_k_user = min(historico_rmse_user, key=historico_rmse_user.get)
    melhor_k_item = min(historico_rmse_item, key=historico_rmse_item.get)

    print(f"\n Melhor KNN User-Based: k={melhor_k_user} (RMSE: {historico_rmse_user[melhor_k_user]:.4f})")
    print(f" Melhor KNN Item-Based: k={melhor_k_item} (RMSE: {historico_rmse_item[melhor_k_item]:.4f})")

    return melhor_k_user, melhor_k_item


def otimizar_svd(treino_puro, base_validacao, valores_reg, fator_fixo=10):
    print("\n=====================================================================")
    print(f"Iniciando a Hiperparametrização do SVD (Fatores Fixos em {fator_fixo})")
    
    historico_rmse_svd = {}

    for reg_atual in valores_reg:
        print(f"\n--- Testando SVD com regularização (reg) = {reg_atual} ---")
        
        # Instancia e treina o modelo 
        model = SVD(k_factors=fator_fixo, reg=reg_atual, epochs=20) 
        model.train(treino_puro)
        
        lista_prev_svd = []
        lista_real_svd = []
        
        # Uso da base de Validação
        for index, linha in base_validacao.iterrows():
            usuario_alvo = int(linha['UserID'])
            filme_alvo = int(linha['MovieID'])
            nota_real = linha['Rating']
            
            try:
                prev = model.predict(usuario_alvo, filme_alvo)
                lista_prev_svd.append(prev)
                lista_real_svd.append(nota_real)
            except IndexError:
                # Caso seja um usuário que só existe na validação e não no treino
                pass 

        # Calcula e guarda o RMSE desta regularização
        rmse_atual = rmse(np.array(lista_prev_svd), np.array(lista_real_svd))
        historico_rmse_svd[reg_atual] = rmse_atual
        print(f"Resultado SVD (reg={reg_atual}) -> RMSE: {rmse_atual:.4f}")

    melhor_reg = min(historico_rmse_svd, key=historico_rmse_svd.get)
    print(f"\nMelhor SVD encontrado: reg={melhor_reg} (RMSE: {historico_rmse_svd[melhor_reg]:.4f})")
    
    return melhor_reg