import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/College_Data', index_col=0)

#print(df.head())
#print(df.describe())
print(df.info())

#sns.set_style('whitegrid')
#sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', palette='coolwarm', size=6, aspect=1, fit_reg=False)


# sns.set_style('whitegrid')
# sns.lmplot('Outstate', 'F.Undergrad', data=df, hue='Private', palette='coolwarm', size=6, aspect=1, fit_reg=False)

# sns.set_style('whitegrid')
# g = sns.FacetGrid(data=df, hue='Private', palette='coolwarm', size=6, aspect=2)
# g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)

# sns.set_style('whitegrid')
# g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
# g = g.map(plt.hist, 'Grad.Rate', bins=30, alpha=0.7 )


# print(df[df['Grad.Rate']>100])


df['Grad.Rate']['Cazenovia College']=100

#print(df[df['Grad.Rate']>100])


# sns.set_style('whitegrid')
# g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
# g = g.map(plt.hist, 'Grad.Rate', bins=30, alpha=0.7 )


plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private', axis=1))

#print(kmeans.cluster_centers_)


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)

print(df.head())



from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))


