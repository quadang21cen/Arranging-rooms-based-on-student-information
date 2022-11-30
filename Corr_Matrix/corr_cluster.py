import pandas as pd
import scipy.cluster.hierarchy as spc
import numpy as np

df = pd.read_csv("demo_rs.csv")
#df = pd.DataFrame(my_data)
corr = df.corr().values

pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
labels_order = np.argsort(idx)
print(idx)
print(labels_order)