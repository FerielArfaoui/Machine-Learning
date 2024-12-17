import numpy as np
import pandas as pd
# Définition des données
data = {
    'Individu': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'],
    'Taille': ['Petit', 'Grand', 'Grand', 'Grand', 'Grand', 'Petit', 'Petit', 'Grand'],
    'Nationalité': ['Italien', 'Italien', 'Italien', 'Français', 'Allemand', 'Allemand', 'Allemand', 'Allemand'],
    'Situation_F': ['Cel.', 'Cel.', 'Marié', 'Cel.', 'Cel.', 'Cel.', 'Marié', 'Marié'],
    'Prend_credit': ['Non', 'Non', 'Non', 'Oui', 'Oui', 'Oui', 'Non', 'Non']
}
df = pd.DataFrame(data)
df['Taille'] = df['Taille'].map({'Petit': 0, 'Grand': 1})
df['Nationalité'] = df['Nationalité'].map({'Italien': 0, 'Français': 1, 'Allemand': 2})
df['Situation_F'] = df['Situation_F'].map({'Cel.': 0, 'Marié': 1})
# Définition des nouveaux individus N9 et N10
N9 = np.array([0, 0, 1])  # Petit, Italien, Marié
N10 = np.array([0, 1, 0])  # Petit, Français, Cel.
def distance(x, y):
    return np.sum(np.abs(x - y))
distances_N9 = np.apply_along_axis(lambda x: distance(N9, x), 1, df[['Taille', 'Nationalité', 'Situation_F']])
distances_N10 = np.apply_along_axis(lambda x: distance(N10, x), 1, df[['Taille', 'Nationalité', 'Situation_F']])
# Trouver les indices des k voisins les plus proches
k = 5
indices_N9 = np.argsort(distances_N9)[:k]
indices_N10 = np.argsort(distances_N10)[:k]
# Obtenir les classes majoritaires parmi les k voisins les plus proches
classe_majoritaire_N9 = df.loc[indices_N9, 'Prend_credit'].mode()[0]
classe_majoritaire_N10 = df.loc[indices_N10, 'Prend_credit'].mode()[0]
print("Pour N9 (k=5) :", classe_majoritaire_N9)
print("Pour N10 (k=5) :", classe_majoritaire_N10)