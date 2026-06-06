import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_pricing.csv")
df.head()
print("Pourcentage de valeurs manquantes",(df.isna().sum().sum()/df.size)*100)



plt.hist(df['SalePrice'], bins=30, color='red', rwidth=0.9, label="prix du marché")
plt.title("Distribution des prix du marché")
plt.xlabel("Prix du marché")
plt.ylabel("Distribution")
plt.legend();

#Normalisation Min Max
df['SalePrice'] = (df['SalePrice']-df['SalePrice'].min())/(df['SalePrice'].max() - df['SalePrice'].min())
df['GrLivArea'] = (df['GrLivArea']-df['GrLivArea'].min())/(df['GrLivArea'].max() - df['GrLivArea'].min())

0]:

plt.hist([df['SalePrice'], df['GrLivArea']], bins=30, color=['red', 'orange'], rwidth=0.9, label=["prix du marché", "surface habitable"])
plt.title("Distribution des prix du marché et de la surface habitable")
plt.xlabel("Prix du marché et surface habitable")
plt.ylabel("Distribution")
plt.xticks([])
plt.legend();

plt.hist2d(df['SalePrice'], df['GrLivArea'], cmap='Blues', bins=15)
plt.title("Taux d'apparition par rapport aux prix du marché et de la surface habitable")
plt.xlabel("Surface habitable")
plt.ylabel("Prix du marché")
plt.colorbar();

df=pd.read_csv("house_pricing.csv")

plt.figure(figsize=(20,12))

plt.subplot(121)
plt.scatter(df['SalePrice'], df['GrLivArea'], color='orange',  label="surface habitable")
plt.title("Distribution des prix du marché en fonction de la surface habitable")
plt.ylabel("Surface habitable")
plt.xlabel("Prix du marché")
plt.ylim(0, 4000)
plt.legend()


plt.subplot(122)
plt.scatter(df['SalePrice'], df['TotalBsmtSF'], color='y', label="surface habitable")
plt.title("Distribution des prix du marché en fonction de la surface de la cave")
plt.xlabel("Prix du marché et surface habitable")
plt.ylabel("Surface de la cave")
plt.xlabel("Prix du marché")
plt.ylim(0, 4000)
plt.legend();


plt.figure(figsize=(10,10))
l = list()
for i in df['OverallQual'].unique() :
    l.append(df[df['OverallQual'] == i]['SalePrice'] )
    
#plt.boxplot(l);


df.boxplot(column='SalePrice', by='OverallQual', figsize=(10,10))




plt.figure(figsize=(15,12))

plt.scatter(df['SalePrice'], df['GrLivArea'], c=df['OverallQual'], alpha=0.4, s=df['TotalBsmtSF'],cmap= 'rainbow')
plt.title("Etude de corrélation multivariable des habitations")
plt.ylabel("Surface habitable")
plt.xlabel("Prix du marché")

plt.plot([10000, 200000, 200000, 10000, 10000], [6000, 6000, 7000,7000,6000], c='black')
plt.text(12500, 6200, "taille: surface de la cave")
plt.text(12500, 6500, "couleur: qualité de finition")
plt.colorbar();

