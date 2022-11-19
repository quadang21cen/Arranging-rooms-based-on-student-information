import pandas as pd
import random
from Data_Augmentation import Data_Augmentation

class user_Generator:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.regions = pd.read_csv("pre_processing\\Map_of_regions.csv").to_numpy().flatten()
        self.hob_inter_list = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['hob_inter'].to_numpy())))
        self.Bio_personality = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['Bio_personality'].to_numpy())))
        self.food_drink = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['food_drink'].to_numpy())))
        self.refer_roommate = list(dict.fromkeys(Data_Augmentation.StringAugmentation().augment(self.data['refer_roommate'].to_numpy())))
        self.smoking = self.data['refer_roommate'].to_numpy()
        self.sex = self.data['Sex'].to_numpy()
        
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
        return [random.choice(self.sex),random.choice(self.regions),
                self.gen_infor(self.Bio_personality),self.gen_infor(self.food_drink),
                self.gen_infor(self.hob_inter_list),random.choice(self.smoking),
                self.gen_infor(self.hob_inter_list),random.randrange(1,10),
                random.randrange(1,10)]

    def get_random_hobby(self):
        return random.choice(self.hobby_list)

if __name__ == "__main__":
    gen = user_Generator("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\Student_Ins.csv")
    new_user = gen.gen_user(2000)
    df = pd.DataFrame(new_user, columns = [ 'Sex','Hometown', 'Bio_personality','food_drink',
                                            'hob_inter','smoking','refer_roommate','Cleanliess','Privacy'])
    real_df = pd.read_csv("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\Student_Ins.csv")
    real_df.drop(['Name','Major'],axis=1,inplace=True)
    
    df.append(real_df)
    df = df.sample(frac=1)
    df.to_csv('new_user_pl_real.csv')
    print("FINISH...")