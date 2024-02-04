import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

rawNatLoc = pd.read_csv('./dataIN/NatLocMovements.csv', index_col=0)
rawPopulation = pd.read_csv('./dataIN/PopulationLoc.csv', index_col=0)
labels = list(rawNatLoc.columns.values[1:])

data = rawNatLoc.merge(rawPopulation, right_index=True, left_index=True)[['City', 'CountyCode', 'Population'] + labels]
data.fillna(np.mean(data[labels], axis=0), inplace=True)

# requirement_1
req1 = pd.DataFrame(data.groupby('CountyCode').sum().apply(
    lambda row: (row['LiveBirths'] / (row['Population'] / 1000)) - (row['Deceased'] / (row['Population'] / 1000)),
    axis=1))
req1.columns = ['NaturalIncrease']
req1.to_csv('./dataOUT/Request_1.csv')

# requirement_2
pd.DataFrame(data.set_index('City').groupby('CountyCode').apply(
    lambda df: pd.Series({label: df[label].idxmax() for label in labels}))).to_csv('./dataOUT/Request_2.csv')

# requirement_3
raw34 = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)

x = StandardScaler().fit_transform(raw34)
pd.DataFrame(x, columns=raw34.columns.values).to_csv('./dataOUT/Xstd.csv')

HC = linkage(x, 'ward')
print(HC)

# requirement_4
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

print('junction with max df: ', j)
print('threshold: ', np.round(t, 2))

# requirement_5
plt.figure(figsize=(10, 10))
plt.title('Dendrogram')
dendrogram(HC, labels=raw34.index.values, leaf_rotation=45)
plt.axhline(t, c='r')
plt.show()
