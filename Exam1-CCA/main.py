import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

dataIndustries = pd.read_csv('./dataIN/Industrie.csv', index_col=0)
dataPopulation = pd.read_csv('./dataIN/PopulatieLocalitati.csv', index_col=0)
labels = list(dataIndustries.columns.values[1:])

data = dataIndustries.merge(right=dataPopulation, left_index=True, right_index=True)
data.fillna(np.mean(data[labels], axis=0), inplace=True)

# request 1
data.apply(lambda row: row[labels] / row['Populatie'], axis=1).to_csv('./dataOUT/Request_1.csv')

# request 2
req_2 = data[['Judet'] + labels].groupby('Judet').sum()
req_2['Productie'] = req_2.max(axis=1)
req_2['Industrie'] = req_2.idxmax(axis=1)
req_2[['Industrie', 'Productie']].to_csv('./dataOUT/Request_2.csv')

# request 3
raw = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
indexes = raw.index
production = raw.columns[:4]
consumption = raw.columns[4:]

x = pd.DataFrame(data=StandardScaler().fit_transform(raw[production]), index=indexes, columns=production)
y = pd.DataFrame(data=StandardScaler().fit_transform(raw[consumption]), index=indexes, columns=consumption)

x.to_csv('./dataOUT/Xstd.csv')
y.to_csv('./dataOUT/Ystd.csv')

# request 4
p = x.shape[1]
q = y.shape[1]
m = min(p, q)
modelCCA = CCA(n_components=m)
modelCCA.fit(x, y)
z, u = modelCCA.transform(x, y)

Z = ['Z' + str(i + 1) for i in range(z.shape[1])]
U = ['U' + str(i + 1) for i in range(u.shape[1])]
pd.DataFrame(data=z, index=indexes, columns=Z).to_csv('./dataOUT/Xscores.csv')
pd.DataFrame(data=u, index=indexes, columns=U).to_csv('./dataOUT/Yscores.csv')

# request 5
Rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, p:]

pd.DataFrame(data=Rxz, index=Z, columns=production).to_csv('./dataOUT/Rxz.csv')
pd.DataFrame(data=Ryu, index=U, columns=consumption).to_csv('./dataOUT/Ryu.csv')


# request 6
def bi_plot(xaxis: np.ndarray, yaxis: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.title('Bi plot (z1, u1) / (z2, u2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(xaxis[:, 0], xaxis[:, 1], c='r', label='X')
    plt.scatter(yaxis[:, 0], yaxis[:, 1], c='b', label='Y')
    plt.legend()
    plt.show()


bi_plot(z[:, [0, 1]], u[:, [0, 1]])
