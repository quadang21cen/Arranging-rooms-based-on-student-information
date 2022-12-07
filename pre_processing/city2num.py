import pandas as pd
import difflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class city2num:
    def __init__(self) -> None:
        self.data = pd.read_csv("pre_processing\\Map_of_regions.csv")
        self.data = self.data.to_numpy()

    def to_num(self, city_str):
        sim_str = [fuzz.partial_ratio(city_str,str) for str in self.data.tolist()]
        sim_max = max(sim_str)
        return sim_str.index(sim_max)

    def get_all(self, list):
        res = []
        for str_city in list:
            res.append(self.to_num(str_city))
        return res

    def to_city(self, list):
        res = []
        for code in list:
            res.append(self.data[code][0])
        return res

def fun_ci2co(list):
    data = pd.read_csv("pre_processing\\Map_of_regions.csv").to_numpy().flatten().tolist()
    lis_code = []
    for str_city in list:
        print(str_city)
        lis_code.append(data.index(str_city))
    return lis_code

if __name__ == "__main__":
    # print(fun_ci2co(['Bac Lieu','Kien Giang']))
    C2N = city2num()
    print(C2N.to_city([54,53]))
    
    # print(C2N.to_num("Bạc ieu"))
    print("FINISH")
