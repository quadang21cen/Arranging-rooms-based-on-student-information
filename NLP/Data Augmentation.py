import pandas as pd
import random
file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
file_pd_copy = file_pd.copy()
data_head = file_pd.columns
print(len(file_pd[:]))
print(data_head)
random_number = random.randint(1, len(file_pd[:]) - 1)

frames = [file_pd]
random_number = random.randint(0, len(file_pd[:]) - 1)
print(random_number)
file_pd.sample(frac=random_number, replace=True)
frames.append(file_pd)
result = pd.concat(frames)
result.to_csv("Student_Ins.csv")