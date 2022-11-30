import pandas as pd
import scipy.cluster.hierarchy as spc
import numpy as np
def find_corr(limit, columns, lists):
    df = pd.DataFrame(lists,
                      columns=['Name', 'val'])

    results = dict()
    print(len(columns))
    for k in range(len(columns)):
        temp_list = []
        #print(len(df[column].tolist()))
        for i in range(len(df[columns[k]].tolist())):
            if k == i:
                continue
            if df[columns[k]][i] >= limit:
                temp_list.append(i)
        if columns[k] in results:
            results[str(columns[k])].append(temp_list)
        else:
            results[str(columns[k])] = temp_list
    return results
columns =['Name', 'val']
lst = [1, 2, 3, 4, 5, 6, 7]
lst2 = [11, 22, 33, 44, 55, 66, 77]
results = find_corr(limit=0.7, columns=columns, lists=list(zip(lst, lst2)))
print(results['Name'])
print(len(results['Name']))