from typing_extensions import Self
from cv2 import SimpleBlobDetector_Params
import pandas as pd
from pip import main
from scipy.spatial.distance import cosine
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import linear_kernel
from TF_IDF.TF_IDF import TF_IDF


class RS:

    def __init__(self) -> None:
        self.data = pd.read_csv('C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\student_data.csv')
        self.all_user = self.data.iloc[:,:1].to_numpy().flatten()
        self.SIM_matrix = pd.DataFrame(index=self.all_user,columns=self.all_user)
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

    def cosine_sim(self,vec):
        return linear_kernel(vec, vec)

    def compute_all_corr(self):
        # self.data["Cleanliess"]
        SIM_cp = self.data[["Cleanliess","Privacy"]]
        print(self.corr_euclidean(SIM_cp))
        # vec_food_drink = self.TF_IDF.text2vec(self.data["food_drink"])
        # return vec_food_drink
        #sim_food_drink =  self.corr_euclidean(vec_food_drink)
        # return self.cosine_sim(vec,vec)
        
        #corr_matrix()

if __name__ == "__main__":
    RS = RS()
    RS.compute_all_corr()
    # res = RS.compute_all_corr()
    # print(res[0])
    # print("FINISH...")




    # from sklearn.metrics.pairwise import linear_kernel
    # a = [[1,2,3]]
    # b = [[2,4,5]]
    # cosine_similarity = linear_kernel(a, b)
    # print(cosine_similarity)

    # a = [[1,2,3],[2,4,5]]
    # cosine_similarity = linear_kernel(a, a)
    # print(cosine_similarity)
    # cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(a))])
    # print(cosine_similarity_pd)


    