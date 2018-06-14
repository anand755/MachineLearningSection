import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(type(cancer))
#print(cancer.keys())

#print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
#print(df.head())
print(df.info())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)



# plt.figure(figsize=(8,6))
#
# plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Pricipal Component')



#plt.show()

print(pca.components_)


df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(10,6))
sns.heatmap(data=df_comp, cmap='plasma')

plt.show()



