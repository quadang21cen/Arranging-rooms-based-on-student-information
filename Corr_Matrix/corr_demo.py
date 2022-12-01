import pandas as pd
import scipy.cluster.hierarchy as spc
import numpy as np
import random
def find_corr_csv(csv_path, limit = 0.7, num_people = 3):
    df = pd.read_csv(csv_path)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    columns = df.columns
    results = dict()
    for k in range(len(columns)):
        temp_list = []
        # print(len(df[column].tolist()))
        for i in range(len(df[columns[k]].tolist())):
            if k == i:
                continue
            if df[columns[k]][i] >= limit:
                temp_list.append(i)
        list_chosen = []
        for j in range(int(num_people)):
            if not temp_list:
                break
            chosen = random.choice(temp_list)
            temp_list.remove(chosen)
            list_chosen.append(chosen)
        list_chosen.sort()
        if k in results:
            results[k].append(list_chosen)
        else:
            results[k] = list_chosen
    return results
def find_corr(columns, lists, limit = 0.7, num_people = 3):
    df = pd.DataFrame(lists,
                      columns=columns)

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
        list_chosen = []
        for j in range(int(num_people)):
            if not temp_list:
                break
            chosen = random.choice(temp_list)
            temp_list.remove(chosen)
            list_chosen.append(chosen)
        list_chosen.sort()
        if k in results:
            results[k].append(list_chosen)
        else:
            results[k] = list_chosen
    return results
# columns =['Name', 'val']
# lst = [1, 2, 3, 4, 5, 6, 7]
# lst2 = [11, 22, 33, 44, 55, 66, 77]
# results = find_corr(limit=0.7, columns=columns, lists=list(zip(lst, lst2)))
# print(results['Name'])
# print(len(results['Name']))

result = find_corr_csv("demo_rs.csv", limit = 0.7, num_people = 5)

print(result)