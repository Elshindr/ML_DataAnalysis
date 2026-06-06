import warnings

from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
import pandas as pd

"""
# CLASSIFICATION
# récupération des infos
df_class = pd.read_csv('diabetes.csv')
df_class.head(5)
#print(X)

# variable cible
X = df_class.drop('Outcome', axis = 1)



y =df_class.Outcome
print(y)

# créer les différents jeu: celui entraitment  et d'évaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Modélisation :: Entrainer l'arbre de décision
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42) #Instanciation du modèle
dt_clf.fit(X_train, y_train) #Entrainement de modèle


#Prediction sur les donnees de test
y_pred_test = dt_clf.predict(X_test)
print(y_pred_test[:5])

print(y_test[:5])
print(y_pred_test[:5])

print(y_test[:5])

#Comparaison entre y_test et y_pred_test :
#True :  l'algorithme a prédit la bonne classe,
#False : il s'est trompé

print(y_test[:5] == y_pred_test[:5])

# Evaluation
print("score train : " , dt_clf.score(X_train, y_train))
print("score test : ", dt_clf.score(X_test,y_test))
## le score entre le jeu d'entrainement et le jeu de test baisse. On appelle cela de l’overfitting


## REGRESSION
df_reg = pd.read_csv('carPrice.csv', index_col= 0)
df_reg.head()
df_reg.info()
X = df_reg.drop("price", axis=1)
y = df_reg["price"]

X_train, X_test,y_train, y_test = train_test_split(X, y , test_size = 0.2)
from sklearn.tree import DecisionTreeRegressor

#Instanciation du modele
dt_reg = DecisionTreeRegressor(random_state=42)

#Entrainement du modele
dt_reg.fit(X_train, y_train)

y_pred_test=dt_reg.predict(X_test)
print(y_pred_test.head())
print(y_test.head())

print(y_test[:5] - y_pred_test[:5])
print("score train : " , dt_reg.score(X_train, y_train))
print("score test : ", dt_reg.score(X_test,y_test))


## Arbre de décision classification
import pandas as pd

df = pd.read_csv('titanic.csv', index_col=0)

df.info()

feats = df.drop('Survived', axis = 1)
print(feats)

target = df.Survived

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 3) # profondeur de l'arbre
model.fit(X_train, y_train)

Calculer le score sur le jeu d'entraînement revient à prédire sur les variables explicatives du jeu d'entraînement X_train
et comparer cette prédiction avec y_train.
L'analyse du score sur le jeu d'entraînement et sur le jeu de test permet d'identifier le surapprentissage
ici le score sur le jeu d'entraînement est plus élevé que le score de test


# on modifie les parametres d'entrainement : min_sample_leaf =nb min d'échantillons requis pour une séparation de noeud.
from sklearn.tree import DecisionTreeClassifier
model_min_samples = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 25, random_state=42)
model_min_samples.fit(X_train, y_train)
print('Score sur ensemble train', model_min_samples.score(X_train, y_train))
print('Score sur ensemble test', model_min_samples.score(X_test, y_test))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


#creation de l'arbre
fig, ax = plt.subplots(figsize=(40, 20))

plot_tree(model_min_samples,
          feature_names=[
              'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C',
              'Embarked_Q', 'Embarked_S'
          ],
          class_names=['Died', 'Survived'],
          filled=True,
          rounded=True)

plt.show()

# graph des feature d'importance dans le modele
feat_importances = pd.DataFrame({
    "Variables":feats.columns,
    "Importance": model_min_samples.feature_importances_
}).sort_values(by='Importance', ascending=False)


feat_importances.nlargest(4, "Importance").plot.bar(x="Variables",
                                                    y="Importance",
                                                    figsize=(15, 5),
                                                    color="#4529de");

# regression linéaire
# test visualisation du calcul destimation pour beta1
from widgets import interactive_MSE

interactive_MSE()

from sklearn.datasets import make_regression
"""
import matplotlib.pyplot as plt
# 
X, y = make_regression(n_samples=100,
                       n_features=1,
                       n_informative=1,
                       noise=10,
                       random_state=42)

X = pd.DataFrame(X, columns=["X1"])
y = pd.Series(y)
plt.figure(figsize=(10, 8))
plt.scatter(X, y, color = "#4529de")
plt.title("\n\n Nuage de points \n\n", fontsize=20)
plt.xlabel("X",labelpad=20, fontsize=20)
plt.ylabel("Y", rotation=365, labelpad=20, fontsize=20)
plt.show()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)
# model créé possède les attributs intercept_ et coef_  =='ordonnée à l'origine et au coefficient directeur de la droite du modèle. C'est respectivement  β0
#   et  β1
print(model.intercept_, model.coef_)
# 𝑦̂ =−0.36+44.335×𝑥
pred =model.predict(X_test)
print(pred)
erreur = pred - y_test
print(erreur)

plt.figure(figsize=(10, 8))

plt.scatter(X_test, y_test, color='#4529de')

plt.plot(X_test, pred ,color='#26dbe0')

plt.title("\n\nDroite de régression et nuage de points sur le jeu de test \n\n", fontsize=20)

plt.xlabel("X",labelpad=20, fontsize=20)

plt.ylabel("Y", rotation=360, labelpad=20, fontsize=20)

plt.show()

# metrics
# Solution 1 : avec les calcul

def metrics(errors):
    import numpy as np
    mse = (errors**2).mean()
    rmse = np.sqrt((errors**2).mean())
    mae = np.abs(errors).mean()
    return mse, rmse, mae

print(metrics(erreur))

# Solution 2 : avec la bibliothe metrics

def metrics_scikit_learn(y_test, predictions):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    return mse, rmse, mae

print(metrics_scikit_learn(y_test, pred))

# regression logistique (classification binaire)
import seaborn as sns
import sklearn.metrics as mt
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=200,
                           n_features=1,
                           n_clusters_per_class=1,
                           n_informative=1,
                           n_repeated=0,
                           n_redundant=0,
                           shuffle=True,
                           random_state=42)

X = pd.DataFrame(X, columns=["X"])

y = pd.Series(y)

fig = plt.figure(figsize=(10, 8))

sns.scatterplot(data=X,
                x=X["X"],
                y=y,
                hue=y,
                s=400,
                alpha=0.3,
                palette=["#a329de", "#26dbe0"],
                legend='full')
sns.regplot(data=X, x=X["X"], y=y, logistic=True, ci=None, color="#f1c232")
plt.title("\n\n Nuage de points \n\n", fontsize=20)

plt.xlabel("X", labelpad=20, fontsize=20)

plt.ylabel("Y", rotation=365, labelpad=20, fontsize=20)

plt.show()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

model = LogisticRegression()

model.fit(X_train,y_train)
#(1)

preds = model.predict(X_test)

#(2)

preds_proba = model.predict_proba(X_test)

# On conserve uniquement les probabilités de la classe 1

preds_proba = pd.DataFrame(preds_proba).iloc[:,1]

#(3)

plt.figure(figsize=(10, 8))

plt.scatter(X_test, preds_proba, color="#4529de", s=200,  alpha=0.3)

plt.xlabel("X", fontsize=20, labelpad=20)

plt.ylabel("Prédictions", fontsize=20, labelpad=50, rotation=360)

plt.title("\n\n Nuage de prédictions en forme de sigmoïde \n\n", fontsize=20)

plt.show()
def metrics(y_true, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    return mt.accuracy_score(y_true, pred), mt.precision_score(y_true, pred), mt.recall_score(y_true, pred), mt.confusion_matrix(y_true, pred)
from sklearn.metrics import classification_report

"""retourne toutes les métriques que lon a présentées plus haut et également la performance par classe. 
On est alors en mesure de savoir si le modèle prédit correctement lensemble des classes ou seulement une partie. 
Cette analyse est particulièrement pertinente lorsquil y a un déséquilibre des classes et quune classe est surreprésentée dans le jeu de données. 
Ce déséquilibre donne souvent lieu à des performances plus faibles sur la classe minoritaire."""

print(classification_report(y_test, preds))
print(metrics(y_test, preds))

print(model.intercept_)

print(model.coef_[0])

# odd_ration
"""Dans notre cas, nous avons estimé la valeur de β1 à 3.61. Son odd ratio sera égal à 37. Cela peut s'interpréter ainsi : 
lorsque la variable explicative augmente d'une unité, alors les chances d'appartenir à la classe 1 augmentent de 3600%. 
Plus généralement, si l'odd ratio est supérieur à 1, cela signifie qu'une augmentation d'une unité de la variable explicative augmente les chances d'appartenir à la classe positive. 
S'il est égal à 1, la variable n'a pas d'impact sur la variable cible et finalement s'il est inférieur à 1, 
une augmentation d'une unité de de la variable diminue les chances d'appartenir à la classe positive.

Par exemple si un odd ratio égal à 1.2, cela signifie que les chances d'appartenir à la classe 1 augmentent de 20%. 
Lorsqu'un odd ratio est égal à 2, les chances d'appartenir à la classe 1 augmentent de 100%, un odd ratio de 3 de 200% etc. 
Au contraire, pour un odd ratio égal à 0.2, les chances d'appartenir à la classe 1 diminuent de 20% et ainsi de suite."""
import numpy as np

np.exp(model.coef_[0])

from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [('simple_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('model_logistic', LogisticRegression())])

  pipe.fit(X_train, y_train)

  predictions = pipe.predict(X_test)

  print(pipe.score(X_test, y_test))