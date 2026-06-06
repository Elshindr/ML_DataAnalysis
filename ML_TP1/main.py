import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

df = pd.read_csv('churn.csv')
df.head()
df = df.drop('customerID', axis=1)
print(df.columns)
df.isna().sum()

#print(df.isna().sum(), df.isna()) # nb des valeurs manquantes par colonnes
"""cols_na = df.columns[df.isna().sum() > 0]
nb_na = df[cols_na].isna().sum()
for i, v in nb_na.items():
    print(i, (v*100/len(df)))"""

#print(df['Partner'].value_counts())
#print(df['Dependents'].value_counts())



"""for c in df.columns:
    print( " == Variable ", df[c].value_counts())"""





# traitement post preparatoire
df['Partner'] = df['Partner'].replace('Yess ', 'Yes')
df['Dependents'] = df['Dependents'].replace("?", "No")


df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype('float')

feats = df.drop('Churn', axis=1)
target = df['Churn']

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)


# traitement (variables expl)
# données manquante, remplissage par la moyenne
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train.loc[:,['MonthlyCharges', 'TotalCharges']] = imputer.fit_transform(X_train[['MonthlyCharges', 'TotalCharges']])

X_test.loc[:,['MonthlyCharges', 'TotalCharges']] = imputer.transform(X_test[['MonthlyCharges', 'TotalCharges']])


print(len(X_train.columns), len(X_test.columns))

# standariser les données num qui suivent une loi normale (variables expl)
cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
sc = StandardScaler()
X_train[cols] = sc.fit_transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])

# Encodé les variables textutels (varaible cible)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# Encode les variables textuels (varaible expli)
oneh = OneHotEncoder(drop = 'first', sparse_output=False)
cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
print(len(X_train.columns), len(X_test.columns), len(X_train.loc[:,cat].columns), len(X_train[cat].columns))

X_train.loc[:,cat] = oneh.fit_transform(X_train[cat])
X_test.loc[:,cat] = oneh.transform(X_test[cat])


# Chaîne à rechercher
search_string = '9571-EDEBV'

# Appliquer la recherche à toutes les colonnes
mask = df.apply(lambda row: row.astype(str).str.contains(search_string).any(), axis=1)

# Filtrer les lignes où la chaîne est trouvée
filtered_df_all_cols = df[mask]
print(filtered_df_all_cols)
# modelisation
from sklearn.linear_model import LogisticRegression

reglog = LogisticRegression()
reglog.fit(X_train, y_train)

# metrics regression logistique
print('Score sur ensemble train', reglog.score(X_train, y_train))
print('Score sur ensemble test', reglog.score(X_test, y_test))

from sklearn.metrics import classification_report
y_pred = reglog.predict(X_test)

print(pd.crosstab(y_test, y_pred, rownames=['Realité'], colnames=['Prédiction']))
print(classification_report(y_test, y_pred))


### Arbre de décision
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

print('Score sur ensemble train', clf.score(X_train, y_train))
print('Score sur ensemble test', clf.score(X_test, y_test))
# metrics arbre
y_clf = clf.predict(X_test)
print(classification_report(y_test, y_clf))
print(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

import matplotlib.pyplot as plt
feat_importances = pd.DataFrame(clf.feature_importances_, index= feats.columns, columns= ['Importance'])
feat_importances.sort_values(by='Importance', ascending=False, inplace= True)
feat_importances.plot(kind='bar', figsize=(7,7))

# Autres tests avec que les variables importantes
cols=['MonthlyCharges','TotalCharges','tenure','gender']
X_train_n = X_train[cols]
X_test_n = X_test[cols]

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_n, y_train)

print(clf.score(X_train_n,y_train))
print(clf.score(X_test_n,y_test))

from sklearn import tree

y_pred_n = clf.predict(X_test_n)
print(pd.crosstab(y_test, y_pred_n, rownames=['Attendus'], colnames=['Predictions']))
print(classification_report(y_test, y_pred_n))


# Affichage de l'arbre de décision
from sklearn.tree import plot_tree

clf = tree.DecisionTreeClassifier(random_state=42,max_depth = 3)

clf.fit(X_train_n, y_train)

fig, ax = plt.subplots(figsize=(20, 20))

plot_tree(clf,
          feature_names = ['MonthlyCharges','TotalCharges','tenure','gender'],
          class_names = ['Yes','No'],
          filled = True,
          rounded = True)

plt.show()

## modèle de forêt aléatoire
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Score sur ensemble train', rf.score(X_train, y_train))
print('Score sur ensemble test', rf.score(X_test, y_test))


# distribution normalisé de la variable cible
target.value_counts(normalize=True)


####################### PIPELINE
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Pipeline pour le traitement des valeurs numériques
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
           ("scaler", StandardScaler())]
)

# Traitement des variables catégorielles
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
categorical_transformer = OneHotEncoder(drop = "first", sparse_output=False)


#Column Tranformer pour appliquer les transformations sur certaines colonnes
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

#Pipeline final de regroupement
clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression())]
)

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)

clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))