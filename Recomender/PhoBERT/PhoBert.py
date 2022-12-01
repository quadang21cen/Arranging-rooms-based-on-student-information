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


emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

class PhoBERT:
  def __init__(self):
    self.stopwords = []
    self.v_phobert = None
    self.v_tokenizer = None
  def load_stopwords(self, stopword_path = "NLP\\vietnamese_stopwords.txt"):
    self.stopwords = []
    with open(stopword_path, encoding='utf-8') as f:
        lines = f.readlines()
    self.stopwords=[line.replace("\n","") for line in lines]
    return self.stopwords

  def standardize_data(self, row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = emoji_pattern.sub(r'', row)
    row = row.replace("http://", " ").replace("<unk>", " ")
    # row = row.replace("http://", " ").replace("(", " ").replace("=", " ")\
    #     .replace(".", " ") \
    #     .replace(";", ",").replace("“", " ") \
    #     .replace(":", ",").replace("”", " ") \
    #     .replace('"', ",").replace("'", " ") \
    #     .replace("!", ",").replace("?", " ") \
    #     .replace("-", ",").replace("?", " ") \
    #     .replace("/", ",")
    row = re.sub(r"\s+", " ", row)  # Remove multiple spaces in content
    row = underthesea.text_normalize(row)
    row = row.strip().lower()
    return row

  def load_bert(self, path = "NLP\\PhoBERT\\RM_system_not_mixed__NLP_model\\"):
    self.v_phobert = AutoModel.from_pretrained(path, from_tf=True)
    self.v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

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
    max_len = 0  # Mỗi câu dài tối đa 100 từ
    v_tokenized = [self.make_bert_encode(i_text) for i_text in v_text]
    for i in v_tokenized:
      if len(i) > max_len:
        max_len = len(i)
    #print(v_tokenized)
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

    #v_features = last_hidden_states[0][:, 0, :]
    embeddings = last_hidden_states[0]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    # Turn torch array into numpy array
    mean_pooled = mean_pooled.detach().numpy()
    #print(mean_pooled)
    # print(v_features.shape)
    return mean_pooled
  def text2vec(self, rows):
    self.load_stopwords()
    self.load_bert()
    features = self.make_bert_features(rows)
    return features

if __name__ == '__main__':
  # example text
  import time

  start_time = time.time()
  text = ["Vẽ, coi phim, chơi game",
          "Vẽ, đọc sách, chơi game",
          "Hướng nội thích ở 1 mình, ko thích  đi chơi",
          "ko thích  đi chơi, Hướng nội thích ở 1 mình"
          ]
  # Gọi hàm text2Vec
  instance_PB = PhoBERT()
  features = instance_PB.text2vec(text)
  print(len(features[0]))
  similarity = cosine_similarity([features[0]], [features[0]])
  print("(0,0):",similarity)
  similarity = cosine_similarity([features[0]], [features[1]])
  print("(0,1):",similarity)
  similarity = cosine_similarity([features[0]], [features[2]])
  print("(0,2):",similarity)
  similarity = cosine_similarity([features[0]], [features[3]])
  print("(0,3):",similarity)
  similarity = cosine_similarity([features[1]], [features[2]])
  print("(1,2):",similarity)
  similarity = cosine_similarity([features[2]], [features[0]])
  print("(2,0):",similarity)
  similarity = cosine_similarity([features[2]], [features[1]])
  print("(2,1):",similarity)
  similarity = cosine_similarity([features[2]], [features[3]])
  print("(2,3):",similarity)

  print("--- %s seconds ---" % (time.time() - start_time))
  # So sánh
  # from sklearn.metrics.pairwise import cosine_similarity
  # import pandas as pd
  # print("-----------------------------------------------")
  # print("So sánh giữa các features (các vector của từng câu)")
  # cosine_similarity = cosine_similarity(features, features)
  # cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns=[*range(len(features))])
  # print(cosine_similarity_pd)