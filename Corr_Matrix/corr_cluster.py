import pandas as pd
import scipy.cluster.hierarchy as spc

df = pd.read_csv("demo_rs.csv")
#df = pd.DataFrame(my_data)
corr = df.corr().values

pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

print(idx)