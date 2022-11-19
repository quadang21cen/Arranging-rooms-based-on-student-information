import pandas as pd
import difflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
class city2num:
    def __init__(self) -> None:
        self.data = pd.read_csv("pre_processing\\Map_of_regions.csv",header=None)
        self.data = self.data.to_numpy()

    def to_num(self, city_str):
        sim_str = [fuzz.partial_ratio(city_str,str) for str in self.data.tolist()]
        sim_max = max(sim_str)
        return sim_str.index(sim_max) + 1

    def get_all(self, list):
        res = []
        for str_city in list:
            res.append(self.to_num(str_city))
        return res

def fun_ci2co(list):
    data = pd.read_csv("pre_processing\\Map_of_regions.csv",header=None).to_dict()[0]
    data = {v: k+1 for k, v in data.items()}
    return np.reshape([data[x] for x in list],(-1, 1))

if __name__ == "__main__":
    test = city2num()
    print(test.get_all(['Lai Chau','Phu Tho']))
    print("FINISH")
