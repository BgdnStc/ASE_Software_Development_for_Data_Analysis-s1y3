import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rawGlobal = pd.read_csv('./dataIN/GlobalIndicatorsPerCapita_2021.csv', index_col=0)
rawCountries = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labels = list(rawGlobal.columns.values[1:])

data = rawGlobal.merge(rawCountries, left_index=True, right_index=True).drop('Country_y', axis=1).rename(
    columns={'Country_x': 'Country'})[['Continent', 'Country'] + labels]
data.fillna(np.mean(data[labels], axis=0), inplace=True)

# A1
valueAdded = list(rawGlobal.columns.values[8:])
pd.DataFrame(data[['Country'] + valueAdded].set_index('Country', append=True).sum(axis=1)).to_csv('./dataOUT/Cerinta1.csv')

# A2
pd.DataFrame(data[['Continent'] + labels].groupby('Continent').apply(
    lambda df: pd.Series({ind: np.round(np.std(df[ind]) / np.mean(df[ind]) * 100, 2) for ind in labels}))
 .to_csv('./dataOUT/Cerinta2.csv'))

# B1
x = StandardScaler().fit_transform(data[labels])

pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print('alpha =', alpha)

# B2
scores = C / np.sqrt(alpha)
pd.DataFrame(scores, index=data.index.values, columns=labels).to_csv('./dataOUT/Scoruri.csv')

# B3
plt.figure(figsize=(10, 10))
plt.title('Scoruri')
plt.scatter(scores[:, 0], scores[:, 1])
plt.show()

# C
factorLoadings = pd.read_csv('./dataIN/g20.csv', index_col=0)
communalities = np.cumsum(factorLoadings * factorLoadings, axis=1)
print('factor with the biggest variance:', communalities.sum().idxmax())
