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

if __name__ == "__main__":
    C2N = city2num()
    print(C2N.to_num("ạc ieu"))
    print("FINISH")
