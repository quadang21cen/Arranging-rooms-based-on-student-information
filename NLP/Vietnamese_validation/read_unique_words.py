import pandas as pd
import underthesea
import re
from nltk.corpus import words
file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
print(file_pd.columns)
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace("http://", " ").replace("(", " ").replace("=", " ") \
      .replace(",", " ").replace(".", " ") \
      .replace(";", " ").replace("“", " ") \
      .replace(":", " ").replace("”", " ") \
      .replace('"', " ").replace("'", " ") \
      .replace("!", " ").replace("?", " ") \
      .replace("-", " ").replace("?", " ") \
      .replace("/", " ").replace(")", " ") \
      .replace("+", " ").replace("%", " ")
    row = row.strip().lower()
    return row

features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                        "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
file_pd.columns = features
file_pd = file_pd.drop("Unnamed", axis = 1)

bio_data = [standardize_data(text) for text in file_pd["Bio_personality"].tolist()]
food_drink_data = [standardize_data(text) for text in file_pd["food_drink"].tolist()]
hobby_interests_data = [standardize_data(text) for text in file_pd["hobby_interests"].tolist()]

tokenized_bio_data = [underthesea.word_tokenize(bio_text) for bio_text in bio_data]
tokens_bio = sum(tokenized_bio_data,[])
tokenized_food_drink_data = [underthesea.word_tokenize(food_drink_data_text) for food_drink_data_text in food_drink_data]
tokens_food = sum(tokenized_food_drink_data,[])
tokenized_hobby_interests_data = [underthesea.word_tokenize(hobby_data_text) for hobby_data_text in hobby_interests_data]
tokens_hobby = sum(tokenized_hobby_interests_data,[])

final_tokens = tokens_bio + tokens_food + tokens_hobby

#print(final_tokens)

unique = []
for word in final_tokens:
    if word.lower() not in unique:
        unique.append(word.lower())
#sort

#print(unique)
my_file = open("words.txt", "r", encoding='utf-8')
# reading the file
data = my_file.read()

data_into_list = data.split("\n")

# Them vao tu tieng anh (dung luong lon)
#result = data_into_list + unique + words.words()

# Them vao nhung tu duoc tokenize boi underthesea
result = data_into_list + unique
result.sort()
print(result)
with open("outfile.txt", "w", encoding='utf-8') as outfile:
    outfile.write("\n".join(result))