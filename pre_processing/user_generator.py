import pandas as pd
import random
from Data_Augmentation import Data_Augmentation
import numpy as np
from city2num import city2num


class user_Generator:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        # self.data = self.pre_HW()
        # self.regions = pd.read_csv("pre_processing\\Map_of_regions.csv").to_numpy().flatten()
        self.df_regions = self.gen_number('Hometown')
        self.hob_inter_list = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['hob_inter'].to_numpy())))
        self.Bio_personality = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['Bio_personality'].to_numpy())))
        self.food_drink = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['food_drink'].to_numpy())))
        self.refer_roommate = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['refer_roommate'].to_numpy())))
        self.df_smoking = self.gen_number('smoking')
        self.df_sex = self.gen_number('Sex')
        self.df_Cleanliess = self.gen_number('Cleanliess')
        self.df_Privacy = self.gen_number('Privacy')

    def gen_user(self, size):
        gen_user_list = []
        for i in range(0,size):
            gen_user_list.append(self.random_pick()) 
        return gen_user_list

    def gen_infor(self, random_list):
        list_ran = []
        for i in range(0,random.randrange(1,6)):
            list_ran.append(random.choice(random_list))
        return ' '.join(map(str,list(dict.fromkeys(list_ran))))
        
    def random_pick(self):
        return [np.random.choice(self.df_sex['Sex'].to_numpy(), p=self.df_sex['Percent'].to_numpy()),
                np.random.choice(self.df_regions['Hometown'].to_numpy(), p=self.df_regions['Percent'].to_numpy()),
                self.gen_infor(self.Bio_personality),
                self.gen_infor(self.food_drink),
                self.gen_infor(self.hob_inter_list),
                np.random.choice(self.df_smoking['smoking'].to_numpy(), p=self.df_smoking['Percent'].to_numpy()),
                self.gen_infor(self.refer_roommate),
                np.random.choice(self.df_Cleanliess['Cleanliess'].to_numpy(), p=self.df_Cleanliess['Percent'].to_numpy()),
                np.random.choice(self.df_Privacy['Privacy'].to_numpy(), p=self.df_Cleanliess['Percent'].to_numpy())
                ]

    def get_random_hobby(self):
        return random.choice(self.hobby_list)

    def gen_number(self, col):
        df1 = self.data.groupby(col).count()[["Name"]].rename(columns= {'Name':'Count'})
        df1 = df1.rename_axis(col).reset_index()
        # df3 = pd.concat([df1,pd.DataFrame([[2,1]], columns=[col, 'Count'])])
        df1["Percent"] = df1['Count']/df1["Count"].sum()
        return df1


    def pre_HW(df):
        C2N = city2num()
        city = df['Hometown'].tolist()
        city_code = C2N.get_all(city)
        name_city = C2N.to_city(city_code)
        df['Hometown'] = name_city
        return df
    

if __name__ == "__main__":
    gen = user_Generator("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\FINAL_Data_set_FixHW.csv")
    new_user = gen.gen_user(10000)
    df = pd.DataFrame(new_user, columns = [ 'Sex','Hometown', 'Bio_personality','food_drink',
                                            'hob_inter','smoking','refer_roommate','Cleanliess','Privacy'])

    
    df.append(gen.data)
    df = df.sample(frac=1)
    df.to_csv('pre_processing\Data_Augmentation_10k.csv',encoding='utf-8-sig')
    print("FINISH...")