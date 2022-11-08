# Import các thư viện cần thiết
import os
import torch
import numpy
import re
import underthesea # Thư viện tách từ
import numpy as np


from transformers import AutoTokenizer, AutoModelForMaskedLM # Thư viện BERT
from transformers import AdamW

from tqdm import tqdm  # for our progress bar
# So sánh
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Hàm load model BERT
def load_bert():
    v_phobert = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
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

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("vietnamese_stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw


# Hàm load dữ liệu từ file data_1.csv để train model
def load_data():
    v_text = []
    v_label = []

    with open('data_1.csv', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace("\n","")
        print(line[:-2])
        v_text.append(standardize_data(line[:-2]))
        v_label.append(int(line[-1:].replace("\n", "")))

    print(v_label)
    return v_text, v_label


class MeditationsDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

# Hàm tạo ra bert features
def make_bert_features(v_text):
    global phobert, sw
    v_tokenized = []
    max_len = 100 # Mỗi câu dài tối đa 100 từ
    text_list = []
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        # print("Word segment  = ", line)
        # Tokenize bởi BERT
        text_list.append(line)
    print("Độ dài lớn nhất:", tokenizer.model_max_length)
    inputs = tokenizer(text_list, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
    print(inputs.keys())
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 0) * (inputs.input_ids != 2) * (inputs.input_ids != 1)
    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(
          torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    print(selection[:5])
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    dataset = MeditationsDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    phobert.to(device)
    # activate training mode
    phobert.train()

    # initialize optimizer
    optim = AdamW(phobert.parameters(), lr=5e-5)
    outputs = None
    epochs = 2

    for epoch in range(epochs):
      # setup loop with TQDM and dataloader
      loop = tqdm(loader, leave=True)
      for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = phobert(input_ids, attention_mask=attention_mask,
                        labels=labels)

        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    from transformers import TrainingArguments

    args = TrainingArguments(
      output_dir='out',
      per_device_train_batch_size=16,
      num_train_epochs=2
    )
    from transformers import Trainer

    trainer = Trainer(
      model=phobert,
      args=args,
      train_dataset=dataset
    )
    trainer.train()
    trainer.save_model("Personal_PhoBERT")


print("Chuẩn bị nạp danh sách các từ vô nghĩa (stopwords)...")
sw = load_stopwords()
print("Đã nạp xong danh sách các từ vô nghĩa")

print("Chuẩn bị nạp model BERT....")
phobert, tokenizer = load_bert()
print("Đã nạp xong model BERT.")

# print("Chuẩn bị load dữ liệu....")
# text, label = load_data()
# print("Đã load dữ liệu xong")

text = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
              "Tao thich nhay mua"
              ]

print("Chuẩn bị tạo features từ BERT.....")
features = make_bert_features(text)
print(features)
print("Đã tạo xong features từ BERT")

cosine_similarity = cosine_similarity(features, features)
cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns=[*range(len(features))])
print(cosine_similarity_pd)
cosine_similarity_pd.to_csv("results.csv")
