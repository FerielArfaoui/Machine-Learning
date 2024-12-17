#  modèle Naive Bayes
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
file = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(file, names=names, sep=';')
data.dropna(subset=['class'], inplace=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modele_naive_bayes = MultinomialNB()
# Entraînement du modèle
modele_naive_bayes.fit(X_train, y_train)
# Prédictions sur l'ensemble de test
predictions = modele_naive_bayes.predict(X_test)
# Matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)
print("Matrice de confusion :")
print(conf_matrix)
# Rapport de classification
class_report = classification_report(y_test, predictions)
print("\nRapport de classification :")
print(class_report)
from sklearn.model_selection import cross_val_score
# Calcul de l'accuracy par validation croisée (par exemple, avec 5 folds)
accuracies = cross_val_score(modele_naive_bayes, X, y, cv=5, scoring='accuracy')
# Affichage des nouvelles valeurs d'accuracy
print("Nouvelles valeurs d'accuracy :", accuracies)
# Affichage de la valeur moyenne d'accuracy
print("Valeur moyenne d'accuracy :", accuracies.mean())