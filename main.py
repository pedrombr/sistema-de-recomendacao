#from algoritmos.knn import teste
from importdataset import carregar_dataset_movies, carregar_dataset_user


def main():
    filmes = carregar_dataset_movies()
    usuarios = carregar_dataset_user()

    print(filmes.head())
    print('------------------')
    print(usuarios.head())

if __name__ == "__main__":
    main()
