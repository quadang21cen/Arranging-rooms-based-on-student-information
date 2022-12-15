
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
        if df_path is str: 
            self.data = self.data = pd.read_csv(df_path, encoding='utf-8')
        else:
            self.data = df_path.iloc[:50,:]
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.data.index,columns=self.data.index)
        self.Pho_BERT = PhoBERT()
        self.trans_city = city2num()

    def corr_cosine(self, util_matrix):
        return cosine_similarity(util_matrix,util_matrix)

    def PhoB2vec(self,data):
        self.Pho_BERT.loadAll()
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
        for index,Val in enumerate(lst):
            ls.append([Val,index])
        ls.sort()
        return ls

    def to_Room(self, groups):
        user_room = []
        for num,group in enumerate(groups):
            for user in group:
                user_room.append([user,num])
        user_room = sorted(user_room)
        to_df = pd.DataFrame(columns=['id', 'room'],data=user_room)
        return to_df

    def grouping(self,np_data,size = 3, constract = False):
        np_data = np_data.to_numpy()[:,1:]
        len_data = len(np_data) - 1
        corr = np.fill_diagonal(np_data,0)
        in_room = []
        dorm = []
        for  id, corr in enumerate(np_data):
            room = []
            i = -1
            if id not in in_room:
                max_size = 0
                room.append(id)
                in_room.append(id)
                new_corr = self.enumLs(corr)
                while max_size != size:
                    if i < - len_data:
                        break
                    if new_corr[i][1] not in in_room:
                        room.append(new_corr[i][1])
                        in_room.append(new_corr[i][1])
                        max_size = max_size + 1
                    i = i - 1
                dorm.append(room)
        return dorm


    def compute_all_corr(self, W_hom = 0.1, W_Bio_per=0.2,W_FaD= 0.2, W_hob = 0.2, W_ref = 0.2, W_cp = 0.2, room_size = 3):
        list_city = self.data["Hometown"].tolist()
        CORR_city = self.normalized(self.city_distance(self.trans_city.get_all(list_city)))
        del list_city
        
        self.Pho_BERT.loadAll()

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

        # Cleanliess and Privacy
        VEC_cp = self.normalized(self.data[["Cleanliess","Privacy"]].to_numpy())
        CORR_cp = self.corr_cosine(VEC_cp)

        res = CORR_city*W_hom + CORR_bio*W_Bio_per + CORR_FaD*W_FaD + CORR_hob*W_hob + CORR_Ref*W_ref + CORR_cp*W_cp + CORR_FaD*0.1
        df_corr = pd.DataFrame(data =res ,index=self.data.index,columns=self.data.index)
        # df_corr.to_csv("Corr_Matrix\\new_corr_noname.csv")
        df_corr.to_csv("DEMO.csv")
        return df_corr

    def normalized(self,vec):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(vec)

    def arrange_ROOM(self,weight=[0.1,0.2,0.2,0.2,0,2], split = False, group_size = 3):     # run this funtion to finish the project
        if split is False:
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr)
            return self.to_Room(df_group)
        else:
            template = self.data
            # execute female
            self.data = self.data[self.data['Sex'] == 'Nam']
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr)
            self.to_Room(df_group).to_csv("Result\\Room_result_FEMALE.csv",index = False)

            self.data = template
            female_df = self.data[self.data['Sex'] == 'Nữ']
            df_corr = self.compute_all_corr()
            df_group = self.grouping(df_corr)
            self.to_Room(df_group).to_csv("Result\\Room_result_MALE.csv",index = False)



def run(data):
    male_df = data[data['Sex'] == 'Nam']
    female_df = data[data['Sex'] == 'Nữ']
    Rs_male = RS(male_df)
    RS.arrange_ROOM().to_csv("Result\\Room_result_MALE.csv",index = False)

    Rs_female = RS(female_df)
    Rs_female.arrange_ROOM().to_csv("Result\\Room_result_FEMALE.csv",index = False)
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
        
        

if __name__ == "__main__":
    print("START...")
    import time
    st = time.time()
    data = pd.read_csv("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\FINAL_Data_set_FixHW.csv", encoding='utf-8')
    RS = RS(data)
    res = RS.arrange_ROOM(split = True)
    # res.to_csv("Result\\Room_result.csv",index = False)
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    print("FINISH")
