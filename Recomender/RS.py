from typing_extensions import Self
import pandas as pd
from pip import main
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn import preprocessing
from Bag_Of_Words.Bag_of_Words_model import Bag_Of_Word

class RS:

    def __init__(self) -> None:
        self.data = pd.read_csv('C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_data.csv')
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
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
        

if __name__ == "__main__":
    RS = RS()
    res = RS.compute_all_corr()
    df_corr = RS.SIM_matrix = pd.DataFrame(data =res ,index=RS.all_user,columns=RS.all_user)
    df_corr.to_csv("demo_rs.csv")
    print(res)
    print("FINISH")

    
