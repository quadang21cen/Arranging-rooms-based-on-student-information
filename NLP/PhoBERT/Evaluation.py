from transformers import TFAutoModelForMaskedLM
import re
import emoji
import underthesea
import datasets

def clean_text(corpus):
    clean_corpus = []
    for i in range(len(corpus)):
        word = corpus[i].lower()
        word = text_normalize(word)
        word = re.sub(r"\s+", " ", word) # Remove multiple spaces in content
        # remove punctuation
        #word = re.sub('[^a-zA-Z]', ' ', word)

        # remove digits and special chars
        word = re.sub("(\\d|\\W)+", " ", word)
        clean_corpus.append(word)
    return clean_corpus
# Hàm chuẩn hoá câu
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
def standardize_data(row):
    if not isinstance(row, str):
        return "không"
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
    row = re.sub(r"\s+", " ", row) # Remove multiple spaces in content
    if row == "":
        return "không"
    row = underthesea.text_normalize(row)
    row = row.strip().lower()
    return row
model_checkpoint = "vinai/phobert-base"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
# text = "http:// Tôi làm gì <mask>?"
# print(standardize_data(text))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
import pandas as pd
file_pd = pd.read_csv("combined_csv.csv", encoding='utf-8')
vaild_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
file_pd = file_pd[['Bio_personality', 'food_drink', 'hobby_interests']]
from sklearn.model_selection import train_test_split
file_pd, test_pd = train_test_split(file_pd, test_size=0.4)
vaild_pd = vaild_pd[['Bio_personality ( tính cách cá nhân )', 'food_drink (Đồ ăn thức uống)', 'hobby_interests (sở thích cá nhân)']]
print(file_pd)
print(test_pd)
# file_text_pd = pd.DataFrame(file_pd.values.ravel('F'))
# file_text_pd.columns = ["text"]
file_pd = file_pd.stack().reset_index()
file_pd.columns = ["level_0", "level_1", "text"]
print(file_pd)
test_pd = test_pd.stack().reset_index()
test_pd.columns = ["level_0", "level_1", "text"]
print(test_pd)
vaild_pd = vaild_pd.stack().reset_index()
vaild_pd.columns = ["level_0", "level_1", "text"]

file_pd['text_cleaned'] = list(map(lambda x:standardize_data(x),file_pd['text']))
test_pd['text_cleaned'] = list(map(lambda x:standardize_data(x),test_pd['text']))
vaild_pd['text_cleaned'] = list(map(lambda x:standardize_data(x),vaild_pd['text']))


file_pd["labels"] = file_pd['text_cleaned'].copy()
test_pd["labels"] = test_pd['text_cleaned'].copy()
vaild_pd["labels"] = vaild_pd['text_cleaned'].copy()
print(len(test_pd['text_cleaned']))
def tokenize_function(examples):
    result = tokenizer(examples["text_cleaned"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tdf = pd.DataFrame({'text_cleaned': file_pd['text_cleaned'].tolist(), "labels":file_pd["labels"].tolist()})
tds = datasets.Dataset.from_pandas(tdf)

test_df = pd.DataFrame({'text_cleaned': test_pd['text_cleaned'].tolist(), "labels":test_pd["labels"].tolist()})
test_df = datasets.Dataset.from_pandas(test_df)

valid_df = pd.DataFrame({'text_cleaned': vaild_pd['text_cleaned'].tolist(), "labels":vaild_pd["labels"].tolist()})
valid_df = datasets.Dataset.from_pandas(valid_df)
tds = datasets.DatasetDict({"train":tds, "test": test_df, "valid": valid_df})
tokenized_datasets = tds.map(
    tokenize_function, batched=True, remove_columns=["text_cleaned", "labels"]
)
print(tokenized_datasets)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
data = [tokenized_datasets["train"][i] for i in range(len(tokenized_datasets["train"]))]
# input_ids = [chunk for chunk in data_collator(data)["input_ids"]]
# attention_masks  = tokenized_datasets["train"]["attention_mask"]
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)
tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)
tf_valid_dataset = tokenized_datasets["valid"].to_tf_dataset(
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)
from transformers import create_optimizer, AdamWeightDecay
test_model = TFAutoModelForMaskedLM.from_pretrained("RM_system_NLP_model\\")
num_train_steps = len(tf_train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
test_model.compile(optimizer=optimizer)
import math
eval_results = test_model.evaluate(tf_valid_dataset)
print(f"Perplexity: {math.exp(eval_results):.2f}")