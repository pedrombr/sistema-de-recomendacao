import os
import pandas as pd

def carregar_dataset_rating():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(BASE_DIR, "dataset_movielens", "u.data")

    names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    dataset_notas = pd.read_csv(caminho, sep='\t', names=names, engine='python')

    return dataset_notas

def carregar_dataset_user():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(BASE_DIR, "dataset_movielens", "u.user")

    names = ['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode']
    dataset_users = pd.read_csv(caminho, sep='|', names=names, engine='python', encoding='latin-1')

    return dataset_users

def carregar_dataset_movies():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(BASE_DIR, "dataset_movielens", "u.item")

    names = ['FilmID', 'Title', 'ReleaseDate', 'VideoReleaseDate', 'IMDbURL']
    dataset_movies = pd.read_csv(caminho, sep='|', names=names, engine='python', encoding='latin-1')

    return dataset_movies
#dataset_notas = pd.read_csv("recsys/dataset_movielens/ratings.dat", sep='::',names=names,engine='python')
