import pandas as pd
import numpy as np
import statsmodels.api as sm
%matplotlib inline

import seaborn as sns
simu_norm = np.random.normal(loc=3, scale=0.1, size=1000)
sns.histplot(simu_norm);

print("moyenne est ", simu_norm.mean())
print("ecart type = ", simu_norm.std())
df = pd.read_csv("youtube.csv")
df.head()

print('types des différentes colonnes',df.dtypes)

df["publishedAt"] = pd.to_datetime(df["publishedAt"] )

print('==========')
print('types des différentes colonnes',df.dtypes)
print('modalites des id des categories', df['categoryId'].unique())
print('Fréquences des différentes modalites', df['categoryId'].value_counts(normalize=True))
df_filtre = df[df["categoryId"].isin([24, 10, 20, 27])]
df_filtre.head()

df_filtre["categoryId"].replace({24:'Entertainment', 10:'Music', 20:'Gaming' ,27:'Education'})
df_filtre["categoryId"].unique()
df_filtre.head()

groupby_m = df_filtre.groupby([pd.Grouper(key = 'publishedAt', freq = 'm'), 
                               df_filtre['categoryId']])\
                     .agg({'likes':'mean'}).unstack().fillna(0)

groupby_m.plot(figsize = (20, 4.5), style = 'o-');

# Etant donnée que l'on compare une variable qualitative, soit la catégorie, face à une variable quantitative, le mois,
# un test d'ANOVA sera utilisé ici

# soit H0: il n'y a pas d'effet significatif entre la catégorie et le mois de publication
#soit H1: il y a un effet signification entre la catégorie de vidéo et le mois de publication

df_filtre["month"] =
s_month = pd.Grouper(key='publishedAt', freq ='m')
print(s_month)

# soit s_month un Series avec les mois
s_month

# 
import statsmodels.api
res = statsmodels.formula.api.ols('month ~ categoryId', data = df).fit()
statsmodels.api.stats.anova_lm(res)

# si dans le tableau sortant PR>F est inférieur à 5% alors on peut rejetter l'hypothése 0, et conclure avec l'hypothese 1
# sinon, on rejete H1 en faveur de H0

categories = df_filtre.groupby([pd.Grouper(key = 'publishedAt', freq = 'm'), 
                                df_filtre['categoryId']])\
                      .agg({'video_id':'count'}).unstack().fillna(0)

categories.plot(figsize = (20, 4.5), style = 'o-');
df['likes'].describe()

:

df_max = df[df['likes'] == df['likes'].max()]
print(df_max)

df_num = df.select_dtypes(include=['int', 'float', 'bool'])
df_num.head()

df.corr()