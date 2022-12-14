import torch
import numpy
import re
from underthesea import text_normalize, word_tokenize  # Thư viện tách từ
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
from warnings import filterwarnings
import pandas as pd
import time
filterwarnings('ignore')


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
  def load_stopwords(self, stopword_path = "Recomender/PhoBERT/vietnamese_stopwords.txt"):
    self.stopwords = []
    with open(stopword_path, encoding='utf-8') as f:
        lines = f.readlines()
    self.stopwords=[line.replace("\n","") for line in lines]
    return self.stopwords

  def standardize_data(self, row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    row = emoji_pattern.sub(r'', row)
    row = row.replace("http://", " ").replace("<unk>", " ")
    row = re.sub(r"\s+", " ", row)  # Remove multiple spaces in content
    row = text_normalize(row)
    row = row.strip().lower()
    return row

  def load_bert(self, path = "dung1308/RM_system_not_mixed__NLP_model"):
    self.v_phobert = AutoModel.from_pretrained(path, from_tf=True)
    self.v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

  def make_bert_encode(self, line):
    # Phân thành từng từ
    line = word_tokenize(str(line))
    # Lọc các từ vô nghĩa
    filtered_sentence = [w for w in line if not w in self.stopwords]
    # Ghép lại thành câu như cũ sau khi lọc
    line = " ".join(filtered_sentence)
    line = word_tokenize(line, format="text")
    line = self.standardize_data(line)
    # print("Word segment  = ", line)
    # Tokenize bởi BERT
    encoded_line = self.v_tokenizer.encode(line)
    return encoded_line
  def make_bert_features(self, v_text):
    v_tokenized = [self.make_bert_encode(i_text) for i_text in v_text]
    maxList = max(v_tokenized, key=lambda i: len(i))
    max_len = len(maxList)
    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])

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

    embeddings = last_hidden_states[0]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    # Turn torch array into numpy array
    mean_pooled = mean_pooled.detach().numpy()
    # Size: (number of texts, 768)
    return mean_pooled
  def loadAll(self):
    self.load_stopwords()
    self.load_bert()
  def text2vec(self, rows):
    features = self.make_bert_features(rows)
    return features

if __name__ == '__main__':
  # Gọi hàm text2Vec
  data = pd.read_csv("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information\\pre_processing\\Data_Augmentation_10k.csv")
  pho = PhoBERT()
  vectors = pho.text2vec(data.iloc[1,3])
