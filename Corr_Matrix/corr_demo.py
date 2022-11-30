import pandas as pd
import scipy.cluster.hierarchy as spc
import numpy as np
def find_corr(csv_path, limit):
    df = pd.read_csv(csv_path)
    #df = pd.DataFrame(my_data)
    columns = df.columns.tolist()
    #print(columns)
    columns.pop(0)

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
results = find_corr("demo_rs.csv", limit=0.7)
print(results["Nguyễn Tiến Dũng"])
print(len(results["Lê Phước Toàn"]))