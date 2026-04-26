from algoritmos.knn import knn_user_based
from importdataset import carregar_dataset_movies, carregar_dataset_user


def main():
    teste = knn_user_based()
    print(teste)

if __name__ == "__main__":
    main()
