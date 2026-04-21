import os
import pandas as pd


def carregar_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(BASE_DIR, "..", "dataset_movielens", "ratings.dat")

    names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    dataset_notas = pd.read_csv(caminho, sep='::', names=names, engine='python')

    return dataset_notas


#dataset_notas = pd.read_csv("recsys/dataset_movielens/ratings.dat", sep='::',names=names,engine='python')
