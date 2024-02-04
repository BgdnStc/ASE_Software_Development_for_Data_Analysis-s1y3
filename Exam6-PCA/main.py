import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rawDeathRate = pd.read_csv('./dataIN/Mortalitate.csv', index_col=0)
rawCountries = pd.read_csv('./dataIN/CoduriTariExtins.csv', index_col=0)
labels = list(rawDeathRate.columns.values[:])

data = rawDeathRate.merge(rawCountries, left_index=True, right_index=True)
data.fillna(np.mean(data[labels]), axis=0, inplace=True)

# A1
pd.DataFrame(data[data['RS'] < 0])[['Continent'] + labels].to_csv('./dataOUT/Cerinta1.csv')

# A2
pd.DataFrame(data.groupby('Continent').mean()).to_csv('./dataOUT/Cerinta2.csv')

# B1
x = StandardScaler().fit_transform(data[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print(alpha)

# B2
scores = C / np.sqrt(alpha)
scoresDF = pd.DataFrame(data=scores, index=rawDeathRate.index.values, columns=['C' + str(i + 1) for i in range(C.shape[1])])
scoresDF.index.name = 'Tari'
scoresDF.to_csv('./dataOUT/Scoruri.csv')

# B3
plt.figure(figsize=(10, 10))
plt.scatter(scores[:, 0], scores[:, 1])
plt.title('Scoruri')
plt.show()
