import os
import torch
import numpy
import re
import underthesea # Thư viện tách từ

from transformers import AutoModel, AutoTokenizer # Thư viện BERT
<<<<<<< HEAD
warnings.filterwarnings('ignore')
=======
import warnings
warnings.filterwarnings('ignore')

class PhoBERT_class:
  def __init__(self):
    self.stopwords = []
    self.v_phobert = None
    self.v_tokenizer = None
  def load_stopwords(self, stopword_path):
    self.stopwords = []
    with open(stopword_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        self.stopwords.append(line.replace("\n",""))
    return self.stopwords

  def standardize_data(self, row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
      .replace(";", " ").replace("“", " ") \
      .replace(":", " ").replace("”", " ") \
      .replace('"', " ").replace("'", " ") \
      .replace("!", " ").replace("?", " ") \
      .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

  def load_bert(self, path):
    self.v_phobert = AutoModel.from_pretrained(path)
    self.v_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
  def make_bert_encode(self, line):
    print("Đang xử lý line = ", line)
    # Phân thành từng từ
    line = underthesea.word_tokenize(line)
    # Lọc các từ vô nghĩa
    filtered_sentence = [w for w in line if not w in self.stopwords]
    # Ghép lại thành câu như cũ sau khi lọc
    line = " ".join(filtered_sentence)
    line = underthesea.word_tokenize(line, format="text")
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

    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    # print('padded:', padded[0])
    # print('len padded:', padded.shape)

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = numpy.where(padded == 1, 0, 1)
    # print('attention mask:', attention_mask[0])

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    # print("Padd = ", padded.size())
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
      last_hidden_states = self.v_phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    # print(v_features.shape)
    return v_features

def text2vec_PhoBERT(rows, stopwords = "vietnamese_stopwords.txt", model= "vinai/phobert-base"):
  phobert_instance = PhoBERT_class()
  phobert_instance.load_stopwords(stopwords)
  phobert_instance.load_bert(model)
  # Extract Features
  features = phobert_instance.make_bert_features(rows)
  return features

if __name__ == '__main__':
  # example text
  text = ["Tôi thích đá bóng",
          "Tôi thích đá banh",
          "Tôi thích bơi lội"
          ]
  # Gọi hàm text2Vec
  features = text2vec_PhoBERT(rows = text, stopwords = "NLP\\PhoBERT\\vietnamese_stopwords.txt", model= "vinai/phobert-base")
  print(features)

  # So sánh
  # from sklearn.metrics.pairwise import cosine_similarity
  # import pandas as pd
  # print("-----------------------------------------------")
  # print("So sánh giữa các features (các vector của từng câu)")
  # cosine_similarity = cosine_similarity(features, features)
  # cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns=[*range(len(features))])
  # print(cosine_similarity_pd)
  
