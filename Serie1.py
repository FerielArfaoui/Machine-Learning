# Charger les données
import pandas as pd
filename ='pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names,sep=';')
print(data.head(768))

# afficher les 15 premier lignes
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
peek=data.head(15)
print(peek)


# nbre obs+car
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
nb_car = data.shape[1]
nb_obs = data.shape[0]
print("le nombre de caractéristiques est :", nb_car)
print("le nombre observations est :", nb_obs)


# type de données
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
typed = data.dtypes
print(typed)


# stat descriptive
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass','pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = data.describe()
print(description)


# only calsses
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
class_counts=data.groupby('class').size()
print(class_counts)

 # valeurs équilibrées ou non
import pandas as pd

emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
class_counts = data['class'].value_counts()
is_balanced = (class_counts.min() / class_counts.max()) >= 0.5

if is_balanced:
    print("Les valeurs de classes sont équilibrées.")
else:
    print("Les valeurs de classes ne sont pas équilibrées.")


 #  correlation entre les variable
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass','pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names,sep=';')
pd.set_option('display.width', 100)
correlations= data.corr(method='pearson')
print(correlations)


#  histogramme
import matplotlib.pyplot as plt
import pandas as pd

emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names, sep=';')
data.hist(figsize=(10, 8))
plt.show()

 #  plots
import matplotlib.pyplot as plt
import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names, sep=';')
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(8,6))
plt.show()


 # Vérifier les valeurs nulles

import pandas as pd
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names, sep=';')
null_values = data.isnull().sum()
print(null_values)


 # valeurs  entre 0 et 1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
emplacement = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(emplacement, names=names, sep=';')
features = data.drop('class', axis=1)
labels = data['class']
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
normalized_data['class'] = labels
print(normalized_data.head())



 # les 3 meilleurs valeurs
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names, sep=';')
X = data.iloc[:,0:8]
y = data.iloc[:,8]
best_features = SelectKBest(score_func=chi2, k=3)
fit = best_features.fit(X, y)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([columns, scores], axis=1)
feature_scores.columns = ['Caractéristique', 'Score']
print(feature_scores.nlargest(3, 'Score'))





import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names, sep=';')

X = data.iloc[:,0:8]
y = data.iloc[:,8]
estimator = LogisticRegression()

rfe = RFE(estimator, n_features_to_select=3)
fit = rfe.fit(X, y)

selected_features = pd.DataFrame({'Caractéristique': X.columns, 'Sélectionnée': fit.support_, 'Rang': fit.ranking_})
print(selected_features[selected_features['Sélectionnée']])
