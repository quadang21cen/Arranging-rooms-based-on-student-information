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
        ran_Hometown = random.choice(self.regions)
        ran_cleaness = random.randrange(1,10)
        ran_privacy = random.randrange(1,10)
        ran_hob_inter = self.gen_infor(self.hob_inter_list)
        ran_refer_roommate = self.gen_infor(self.hob_inter_list)
        ran_Bio_personality = self.gen_infor(self.Bio_personality)
        ran_food_drink = self.gen_infor(self.food_drink)
        return [ran_Hometown,ran_Bio_personality,ran_food_drink,ran_hob_inter,random.randrange(0,2),ran_refer_roommate,random.randrange(0,11),random.randrange(0,11)]

    def get_random_hobby(self):
        return random.choice(self.hobby_list)

if __name__ == "__main__":
    gen = user_Generator("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_data.csv")
    new_user = gen.gen_user(10000)
    # print(new_user[0])
    # print(new_user[1])
    # print(new_user[2])
    # print(new_user[3])
    df = pd.DataFrame(new_user, columns = [ 'Hometown', 'Bio_personality','food_drink',
                                            'hob_inter','smoking','refer_roommate','Cleanliess','Privacy'])
                                            
    df.to_csv('argu.csv')
    print("FINISH...")