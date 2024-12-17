# Définition de la matrice de confusion
confusion_matrix = {
    'Don’t Buy-computer': {'Don’t Buy-computer': 407, 'Buy-computer': 101},
    'Buy-computer': {'Don’t Buy-computer': 95, 'Buy-computer': 539}
}

# 1. Calcul des vrais positifs, vrais négatifs, faux positifs et faux négatifs
VP = confusion_matrix['Buy-computer']['Buy-computer']
VN = confusion_matrix['Don’t Buy-computer']['Don’t Buy-computer']
FP = confusion_matrix['Don’t Buy-computer']['Buy-computer']
FN = confusion_matrix['Buy-computer']['Don’t Buy-computer']

# 2. Affichage des résultats
print("Vrais Positifs (VP) :", VP)
print("Vrais Négatifs (VN) :", VN)
print("Faux Positifs (FP) :", FP)
print("Faux Négatifs (FN) :", FN)

# 3. Calcul des mesures d'évaluation
accuracy = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP)

# 4. Affichage des mesures d'évaluation
print("Accuracy :", accuracy)
print("Precision :", precision)
