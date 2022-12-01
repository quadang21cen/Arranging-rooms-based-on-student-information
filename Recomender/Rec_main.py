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

class RS:

    def __init__(self, path = 'C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_ins.csv') -> None:
        self.data = pd.read_csv(path)
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.data.index,columns=self.data.index)
        self.Pho_BERT = PhoBERT()
        self.trans_city = city2num()

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

    def compute_all_corr(self):
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
        # Cleanliess and Privacy
        VEC_cp = self.normalized(self.data[["Cleanliess","Privacy"]].to_numpy())
        CORR_cp = self.corr_cosine(VEC_cp)

        res = CORR_city*0.1 + CORR_bio*0.2 + CORR_hob*0.2 + CORR_Ref*0.2 + CORR_cp*0.3
        df_corr = pd.DataFrame(data =res ,index=self.data.index,columns=self.data.index)
        df_corr.to_csv("Corr_Matrix\\corr_noname.csv")
        return res

    def normalized(self,vec):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(vec)
        

if __name__ == "__main__":

    RS = RS()
    res = RS.compute_all_corr()
    print("FINISH")

    
