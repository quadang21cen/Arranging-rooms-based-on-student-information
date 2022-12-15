
from typing_extensions import Self
import pandas as pd
from pip import main
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn import preprocessing
from PhoBERT.PhoBert import PhoBERT
from city2num import *
from Vietnamese_validation.Vietnamese_validation import isMeaning
import sys
sys.path.append('c:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information\\Recomender\\PhoBERT')


class RS:
    def __init__(self, df_path) -> None:
        if type(df_path) is str: 
            self.data = self.data = pd.read_csv(df_path, encoding='utf-8')
        else:
            self.data = df_path
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.data.index,columns=self.data.index)
        self.Pho_BERT = PhoBERT()
        self.trans_city = city2num()
        self.ID = self.data.iloc[:,0]
        self.current_ROOM = 0
    def corr_cosine(self, util_matrix):
        return cosine_similarity(util_matrix,util_matrix)

    def PhoB2vec(self,data):
        vec = self.Pho_BERT.text2vec(data)
        vec = self.normalized(vec)
        return vec

    # this function will check and if text input is not meanning on of this row will equal 1
    def check_text(self, corr_matrix, list_text, set_value = 0.9):
        for i, text in enumerate(list_text):
            if isMeaning(text) == False:
                corr_matrix[i,:] = set_value
                corr_matrix[:,i] = set_value
        return corr_matrix
    
    def city_distance(self,data):
        corr_rs = []
        for i in data:
            row = []
            for j in data:
                row.append(abs(i - j))
            corr_rs.append(row)
        return corr_rs

    def enumLs(self, lst):
        ls = []
        for index,Val in zip(self.ID,lst):
            ls.append([Val,index])
        ls.sort()
        return ls

    def to_Room(self, groups,start_with = 5):
        user_room = []
        for group in groups:
            for user in group:
                user_room.append([user,start_with])
            start_with = start_with + 1
        user_room = sorted(user_room)
        to_df = pd.DataFrame(columns=['id', 'room'],data=user_room)
        self.current_ROOM = start_with
        return to_df

    def grouping(self,np_data,max_size = 4, constract = False):
        len_data = len(np_data) - 1
        in_room = []
        dorm = []
        for  id, corr in zip(self.ID, np_data):
            room = [] 
            if id not in in_room:
                curr_size = 1
                room.append(id)
                in_room.append(id)
                new_corr = self.enumLs(corr)
                i = -1
                while curr_size < max_size:
                    if new_corr[i][1] not in in_room:
                        room.append(new_corr[i][1])
                        in_room.append(new_corr[i][1])
                        curr_size = curr_size + 1
                    if i == - len_data:        
                        break
                    i = i - 1
                dorm.append(room)
        print(dorm)
        return dorm


    def compute_all_corr(self, W_hom = 0.1, W_Bio_per=0.2,W_FaD= 0.2, W_hob = 0.2, W_ref = 0.2, W_cp = 0.2, room_size = 3):
        list_city = self.data["Hometown"].tolist()
        CORR_city = self.normalized(self.city_distance(self.trans_city.get_all(list_city)))
        del list_city

        # Bio_personality
        bio = self.data["Bio_personality"].to_list()
        VEC_bio = self.Pho_BERT.text2vec(bio)
        CORR_bio = self.check_text(self.corr_cosine(VEC_bio),bio) 
        del VEC_bio, bio

        # hob_inter
        hob = self.data["hob_inter"]
        VEC_hob = self.Pho_BERT.text2vec(hob)
        CORR_hob = self.check_text(self.corr_cosine(VEC_hob),hob)
        del VEC_hob, hob

        #Refer roommate
        ref = self.data["refer_roommate"]
        Vec_ref = self.Pho_BERT.text2vec(ref)
        CORR_Ref = self.check_text(self.corr_cosine(Vec_ref),ref)
        del Vec_ref, ref

        #food_drink
        FaD = self.data["food_drink"]
        Vec_FaD = self.Pho_BERT.text2vec(FaD)
        CORR_FaD = self.check_text(self.corr_cosine(Vec_FaD),FaD)
        return CORR_FaD
        Cleanliess and Privacy
        VEC_cp = self.normalized(self.data[["Cleanliess","Privacy"]].to_numpy())
        CORR_cp = self.corr_cosine(VEC_cp)
        # return CORR_cp

        res = CORR_city*W_hom + CORR_bio*W_Bio_per + CORR_FaD*W_FaD + CORR_hob*W_hob + CORR_Ref*W_ref + CORR_cp*W_cp + CORR_FaD*0.1
        df_corr = pd.DataFrame(data =res ,index=self.data.index,columns=self.data.index)
        # df_corr.to_csv("Corr_Matrix\\new_corr_noname.csv")
        df_corr.to_csv("DEMO.csv")
        return df_corr

    def normalized(self,vec):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(vec)

    def arrange_ROOM(self,weight=[0.1,0.2,0.2,0.2,0.1,0,1], split_gender = False, room_size = 3):     # run this funtion to finish the project
        if split_gender is False:
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr,max_size=room_size)
            to_ROOM = self.to_Room(df_group,start_with=1)
            to_ROOM.to_csv("Result\\Room_result_split_FALSE.csv",index = False)
            return to_ROOM
        else:
            template = self.data
            self.data = self.data[self.data['Sex'] == 'Nữ']
            self.ID = self.data.iloc[:,0]
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr,max_size=room_size)
            ROOM_female = self.to_Room(df_group,start_with= 5)
            # ROOM_female.to_csv("Result\\Room_result_FEMALE.csv",index = False)

            self.data = template[template['Sex'] == 'Nam']
            self.ID = self.data.iloc[:,0]
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr,max_size=room_size)
            ROOM_male = self.to_Room(df_group,start_with= self.current_ROOM)
            # ROOM_male.to_csv("Result\\Room_result_MALE.csv",index = False)
            final_ROOM = pd.concat([ROOM_male,ROOM_female]).sort_values('id')
            final_ROOM.to_csv("Result\\Room_result_split_TRUE.csv")
            
            print(final_ROOM)
            return final_ROOM
            # final_ROOM.to_csv("Result\\Room_result.csv")


            # # execute female
            # self.data_female = self.data[self.data['Sex'] == 'nữ']
            # df_corr = self.compute_all_corr()
            # df_group = self.grouping(df_corr,max_size=room_size)
            # ROOM_female = self.to_Room(df_group)         # to_csv("Result\\Room_result_FEMALE.csv",index = False)
            # ROOM_female.to_csv("Result\\Room_result_FEMALE.csv",index = False)
            # # print(ROOM_female)
                     # to_csv("Result\\Room_result_FEMALE.csv",index = False)
            # # print("--------------")
            # # print(ROOM_male)
            # ROOM_male.to_csv("Result\\Room_result_MALE.csv",index = False)
            # final_ROOM = ROOM_female.add(ROOM_male)
            # print(final_ROOM)
            # final_ROOM.to_csv("Result\\Room_result.csv")


            # room_female = self.to_Room(df_group)
            # self.data = template
            # female_df = self.data[self.data['Sex'] == 'Nữ']
            # df_corr = self.compute_all_corr()
            # df_group = self.grouping(df_corr)
            # room_male = self.to_Room(df_group)
            
            # res = room_female.add(room_male, fill_value=0)
            # res = res.sort_values('id')
            # print(res)
            # self.to_Room(df_group).to_csv("Result\\Room_result_MALE.csv",index = False)
def replace_zero(ls_weight):
    count_zero = 0
    count_not_zero = 0
    for i in range(0,len(ls_weight)):
        if ls_weight[i] == '':
            ls_weight[i] = 0
            count_zero = count_zero + 1
        else: 
            count_not_zero = float(ls_weight[i]) + count_not_zero
            ls_weight[i] = float(ls_weight[i])
    replace = (1 - count_not_zero)/count_zero
    for i in range(0,len(ls_weight)):
        if ls_weight[i] == 0:
            ls_weight[i] = replace
    return ls_weight

def more_than_one(ls_weight):
    all_value = float(0)
    for i in range(0,len(ls_weight)):
        if ls_weight[i] != '':
            all_value = float(ls_weight[i]) + all_value
    if all_value > 1 or all_value < 0:
        return False
    return True

def run(data):
    male_df = data[data['Sex'] == 'Nam']
    female_df = data[data['Sex'] == 'Nữ']
    Rs_male = RS(male_df)
    file_male = Rs_male.arrange_ROOM()
    file_male.to_csv("file.csv")
    # file_male.to_csv("Result\\Room_result_MALE.csv",index = False)

    # Rs_female = RS(female_df)
    # Rs_female.arrange_ROOM().to_csv("Result\\Room_result_FEMALE.csv",index = False)
        

if __name__ == "__main__":
    print("START...")
    import time
    st = time.time()
    data = pd.read_csv("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\FINAL_Data_set_FixHW.csv", encoding='utf-8')
    RS = RS(data)
    res = RS.arrange_ROOM(split_gender = True)
    
    




    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print("FINISH")