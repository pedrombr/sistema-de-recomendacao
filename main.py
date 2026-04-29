from algoritmos.knn import knn_user_based, knn_item_based
from importdataset import carregar_dataset_movies, carregar_dataset_user


def main():
    usuario_teste = 10

    recomendacoes = knn_user_based(user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)
    print("User based")
    if isinstance(recomendacoes, str): # Caso retorne mensagem de erro
        print(recomendacoes)
    else:
        print("\n Top Filmes Recomendados:")
        for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes, 1):
            print(f"{posicao}º Lugar | Filme ID: {filme_id}")

    print("---------------------------------------------------------------------")
    print("Item based")
    recomendacoes_item = knn_item_based(user_id=usuario_teste, k_vizinhos=5, n_recomendacoes=5, normalizar=False)

    if isinstance(recomendacoes_item, str): # Caso retorne mensagem de erro
        print(recomendacoes_item)
    else:
        print("\n Top Filmes Recomendados:")
        for posicao, (filme_id, nota_prevista) in enumerate(recomendacoes_item, 1):
            print(f"{posicao}º Lugar | Filme ID: {filme_id}")

#| Nota Prevista: {nota_prevista:.2f}
if __name__ == "__main__":
    main()
