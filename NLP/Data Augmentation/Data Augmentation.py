import pandas as pd
import random
import re
import string
file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
file_pd_copy = file_pd.copy()
data_head = ['Bio_personality ( tính cách cá nhân )', 'food_drink (Đồ ăn thức uống)',
       'hobby_interests (sở thích cá nhân)']
print(len(file_pd[:]))
print(data_head)
def swap_random(seq):
     idx = range(len(seq))
     i1, i2 = random.sample(idx, 2)
     seq[i1], seq[i2] = seq[i2], seq[i1]
     return seq
frames = [file_pd]
epoch = 1
def get_random_string():
    # With combination of lower and upper case
    N = random.randint(5, 15)
    result_str = ''.join(random.choices(string.ascii_letters, k=N))
    # print random string
    print(result_str)
    return result_str

# Main function
def main(file_pd, name_column, columns, epoch = 2):
    file_pd_copy = file_pd.copy()
    for _ in range(epoch):
        # Name (Tên) sẽ đc thay thế bằng từ ngẫu nhiên
        for i in range(len(file_pd_copy[name_column])):
            file_pd_copy[name_column] = get_random_string()
        # Những feature (chứa string) sẽ được hoán đổi và ngẫu nhiên có thể xóa bớt một token trong cây
        for feature in columns:
            for i in range(len(file_pd_copy[feature])):
                tokenize = (file_pd_copy[feature][i].strip()).split(",")
                tokenize_copy = tokenize.copy()
                for i in range(len(tokenize_copy)):
                    if "và" in tokenize_copy[i]:
                        words = tokenize_copy[i].split("và")
                        tokenize.remove(tokenize_copy[i])
                        tokenize = [*tokenize, *words]
                random.shuffle(tokenize)
                delete_gate = random.uniform(0, 1)
                if delete_gate >= 0.5 and len(tokenize) > 1:
                    del tokenize[0]
                text = ', '.join(tokenize)
                text = re.sub(' +', ' ', text)  # Remove more than one space
                file_pd_copy[feature][i] = text
                print(text)
                # print(file_pd_copy[feature])

        frames.append(file_pd_copy)
    result = pd.concat(frames)
    return result

# Main run in current file
for _ in range(epoch):
    for i in range(len(file_pd_copy["Name (Họ và tên)"])):
        file_pd_copy["Name (Họ và tên)"] = get_random_string()
    for feature in data_head:
        for i in range(len(file_pd_copy[feature])):
            tokenize = (file_pd_copy[feature][i].strip()).split(",")
            tokenize_copy = tokenize.copy()
            for i in range(len(tokenize_copy)):
                if "và" in tokenize_copy[i]:
                    words = tokenize_copy[i].split("và")
                    tokenize.remove(tokenize_copy[i])
                    tokenize = [*tokenize, *words]
            random.shuffle(tokenize)
            delete_gate = random.uniform(0, 1)
            if delete_gate >= 0.5 and len(tokenize) > 1:
                del tokenize[0]
            text = ', '.join(tokenize)
            text = re.sub(' +', ' ',text) # Remove more than one space
            file_pd_copy[feature][i] = text
            print(text)
            #print(file_pd_copy[feature])

    frames.append(file_pd_copy)
result = pd.concat(frames)
result.to_csv("Student_Ins.csv")