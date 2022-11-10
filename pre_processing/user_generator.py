import pandas as pd
import random


class user_Generator:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.regions = pd.read_csv("Map_of_regions.csv")
        
    def gen_user(self, size):
        df_new_users = pd.DataFrame(columns = self.data.columns,data=self.data)
        all_hometown_data = 

    def get_Random_Regions()


if __name__ == "__main__":
    gen = user_Generator("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_data.csv")
    print(random.randrange(3, 9))
    print("FINISH...")