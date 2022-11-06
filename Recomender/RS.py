from typing_extensions import Self
import pandas as pd
from pip import main
import numpy as np
from numpy.linalg import norm
<<<<<<< HEAD

from TF_IDF.TF_IDF import TF_IDF
=======
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn import preprocessing
from Bag_Of_Words.Bag_of_Words_model import Bag_Of_Word
>>>>>>> 268092a8f819ca591f4f026ee1882feef41dd756

class RS:

    def __init__(self) -> None:
        self.data = pd.read_csv('C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_data.csv')
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
<<<<<<< HEAD
        self.TF_IDF = TF_IDF()

    def corr_cosine(self, util_matrix):
        SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
        for i in range(0,util_matrix.__len__()):
            for j in range(0,util_matrix.__len__()):
                SIM_matrix.iloc[i][j] = cosine(util_matrix.iloc[i,1:].to_numpy(),util_matrix.iloc[j,1:].to_numpy())
        return SIM_matrix

    def corr_euclidean(self, util_matrix):
        SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
        for i in range(0,util_matrix.__len__()):
            for j in range(0,util_matrix.__len__()):
                SIM_matrix.iloc[i][j] = np.linalg.norm(util_matrix.iloc[i,1:].to_numpy() - util_matrix.iloc[j,1:].to_numpy())
        return SIM_matrix

    def cosine_sim(self,A , B):
        return np.dot(A,B)/(norm(A, axis=1)*norm(B))

    def corr_matrix(self,col):
        return self.corr_cosine(self.data[col])

    def compute_all_corr(self):
        vec_food_drink = self.TF_IDF.text2vec(self.data["food_drink"])
        return vec_food_drink
        #sim_food_drink =  self.corr_euclidean(vec_food_drink)
        # return self.cosine_sim(vec,vec)
        
        #corr_matrix()
=======
        self.BOW = Bag_Of_Word()

    def corr_cosine(self, util_matrix):
        return cosine_similarity(util_matrix,util_matrix)

    # def corr_euclidean(self, util_matrix):
    #     SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
    #     for i in range(0,util_matrix.__len__()):
    #         for j in range(0,util_matrix.__len__()):
    #             SIM_matrix.iloc[i][j] = np.linalg.norm(util_matrix.iloc[i,1:].to_numpy() - util_matrix.iloc[j,1:].to_numpy())
    #     return SIM_matrix

    def compute_all_corr(self):
        # Cleanliess and Privacy
        VEC_cp = self.data[["Cleanliess","Privacy"]].to_numpy()
        VEC_cp = self.normalized(VEC_cp)
        SIM_cp = self.corr_cosine(VEC_cp)

        # food and drink
        VEC_fd = self.BOW.text2vec(self.data["food_drink"])
        SIM_fd = self.corr_cosine(VEC_fd)
        
        # Bio_personality
        VEC_bp = self.BOW.text2vec(self.data["Bio_personality"])
        SIM_bp = self.corr_cosine(VEC_bp)
        
        # hob_inter
        VEC_hi = self.BOW.text2vec(self.data["hob_inter"])
        SIM_hi = self.corr_cosine(VEC_hi)
        res = (SIM_cp + SIM_fd + SIM_bp + SIM_hi)/4

        return res
    def normalized(self,vec):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(vec)
        
>>>>>>> 268092a8f819ca591f4f026ee1882feef41dd756

if __name__ == "__main__":
    RS = RS()
    res = RS.compute_all_corr()
<<<<<<< HEAD
    print(res[0])
    print("FINISH...")
=======
    df_corr = RS.SIM_matrix = pd.DataFrame(data =res ,index=RS.all_user,columns=RS.all_user)
    df_corr.to_csv("demo_rs.csv")
    print(res)
    print("FINISH")

    
>>>>>>> 268092a8f819ca591f4f026ee1882feef41dd756
