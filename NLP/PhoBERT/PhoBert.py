import os
import torch
import numpy
import re
import underthesea # Thư viện tách từ
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
import unicodedata
import warnings
warnings.filterwarnings('ignore')

class PhoBERT_class:
  def __init__(self):
    self.stopwords = []
    self.v_phobert = None
    self.v_tokenizer = None
  def load_stopwords(self, stopword_path = "vietnamese_stopwords.txt"):
    self.stopwords = []
    with open(stopword_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        self.stopwords.append(line.replace("\n",""))
    return self.stopwords

  def standardize_data(self, row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xoa dau cach khong can thiet
    row = re.sub(' +', ' ', row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
      .replace(";", " ").replace("“", " ") \
      .replace(":", " ").replace("”", " ") \
      .replace('"', " ").replace("'", " ") \
      .replace("!", " ").replace("?", " ") \
      .replace("-", " ").replace("?", " ")

    row = row.strip().lower()
    return row

  def load_bert(self, path = "vinai/phobert-base"):
    self.v_phobert = AutoModel.from_pretrained(path)
    self.v_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

  def make_bert_encode(self, line):
    # Phân thành từng từ
    line = underthesea.word_tokenize(line)
    # Lọc các từ vô nghĩa
    filtered_sentence = [w for w in line if not w in self.stopwords]
    # Ghép lại thành câu như cũ sau khi lọc
    line = " ".join(filtered_sentence)
    line = underthesea.word_tokenize(line, format="text")
    line = self.standardize_data(line)
    # print("Word segment  = ", line)
    # Tokenize bởi BERT
    encoded_line = self.v_tokenizer.encode(line)
    return encoded_line
  def make_bert_features(self, v_text):
    v_tokenized = []
    max_len = 100  # Mỗi câu dài tối đa 100 từ
    for i_text in v_text:
      line = self.make_bert_encode(i_text)
      v_tokenized.append(line)
    print(v_tokenized)
    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    # print('padded:', padded[0])
    # print('len padded:', padded.shape)

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = numpy.where(padded == 1, 0, 1)
    # print('attention mask:', attention_mask[0])

    # Chuyển thành tensor
    padded = torch.tensor(padded).long()
    # print("Padd = ", padded.size())
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
      last_hidden_states = self.v_phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    # print(v_features.shape)
    return v_features
  def text2vec_PhoBERT(self, rows):
    self.load_stopwords()
    self.load_bert()
    features = self.make_bert_features(rows)
    return features

if __name__ == '__main__':
  # example text
  text = ["Vẽ, coi phim, chơi game",
          "Vẽ, đọc sách, chơi game",
          "Hướng nội thích ở 1 mình, ko thích  đi chơi"
          ]
  # Gọi hàm text2Vec
  instance_PB = PhoBERT_class()
  features = instance_PB.text2vec_PhoBERT(text)
  print(len(features[0]))
  similarity = cosine_similarity([features[0]], [features[1]])
  print(similarity)
  similarity = cosine_similarity([features[0]], [features[2]])
  print(similarity)
  # So sánh
  # from sklearn.metrics.pairwise import cosine_similarity
  # import pandas as pd
  # print("-----------------------------------------------")
  # print("So sánh giữa các features (các vector của từng câu)")
  # cosine_similarity = cosine_similarity(features, features)
  # cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns=[*range(len(features))])
  # print(cosine_similarity_pd)