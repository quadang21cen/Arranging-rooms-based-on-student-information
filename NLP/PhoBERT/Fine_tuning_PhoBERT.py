from transformers import TFAutoModelForMaskedLM
import re
import underthesea
import datasets

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
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

num_train_steps = len(tf_train_dataset)
# optimizer = AdamWeightDecay(lr=2e-5, weight_decay_rate=0.01)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")
model_name = model_checkpoint.split("/")[-1]
push_to_hub_model_id = f"{model_name}-finetuned-vbert"
callback = PushToHubCallback(
    output_dir="./RM_system_not_mixed__NLP_model", tokenizer=tokenizer
)
model.compile(optimizer=optimizer)
history = model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=20, callbacks=[callback])

import math
eval_results = model.evaluate(tf_valid_dataset)
print(f"Perplexity: {math.exp(eval_results):.2f}")

with open('Perplexity.txt', 'w') as f:
    f.write(f"Perplexity of RM_system_NLP_not_mixed_float16_model: {math.exp(eval_results):.2f}")
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_not_mixed_float16.png')
# import numpy as np
# import tensorflow as tf

# inputs = tokenizer(text, return_tensors="np")
# token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# # print(inputs["input_ids"])
# # print(tokenizer.mask_token_id)
# # Mask number: 64000
# mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# # We negate the array before argsort to get the largest, not the smallest, logits
# top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()
#
# for token in top_5_tokens:
#     print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")

# def tokenize_function(examples):
#     result = tokenizer(examples[",food_drink"])
#     #print(result)
#     #print(tokenizer.is_fast)
#     if tokenizer.is_fast:
#         result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
#     return result
# import pandas as pd
# filename = 'hobby_interests.csv'
# data = pd.read_csv(filename, sep="\t", encoding='utf-8')
# print(data.columns)
# print(data)
# from sklearn.model_selection import train_test_split
# import datasets
# train_test_data, unsupervised_data = train_test_split(data, test_size=0.5)
# train_dataset, test_dataset= train_test_split(train_test_data, test_size=0.5)
# train_dataset = datasets.Dataset.from_dict(train_dataset)
# test_dataset = datasets.Dataset.from_dict(test_dataset)
# unsupervised_dataset = datasets.Dataset.from_dict(unsupervised_data)
# dd = datasets.DatasetDict({"train":train_dataset,"test":test_dataset, 'unsupervised':unsupervised_dataset})
# print(dd)
# # tokenized_datasets = tokenize_function(data[",food_drink"].to_list())
# tokenized_datasets = dd.map(
#     tokenize_function, batched=True, remove_columns = [",food_drink"]
# )
# print(tokenized_datasets)
#
# print(tokenizer.model_max_length)
#
# chunk_size = 56
# # Slicing produces a list of lists for each feature
# tokenized_samples = tokenized_datasets["train"][:20]
#
# for idx, sample in enumerate(tokenized_samples["input_ids"]):
#     print(f"'>>> Review {idx} length: {len(sample)}'")
#
# concatenated_examples = {
#     k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
# }
# total_length = len(concatenated_examples["input_ids"])
# print(f"'>>> Concatenated reviews length: {total_length}'")
#
# chunks = {
#     k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#     for k, t in concatenated_examples.items()
# }
#
# for chunk in chunks["input_ids"]:
#     print(f"'>>> Chunk length: {len(chunk)}'")
#
# def group_texts(examples):
#     # Concatenate all texts
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     # Compute length of concatenated texts
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the last chunk if it's smaller than chunk_size
#     total_length = (total_length // chunk_size) * chunk_size
#     # Split by chunks of max_len
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     # Create a new labels column
#     result["labels"] = result["input_ids"].copy()
#     return result
#
# # text_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1,
# #     num_proc=4)
# # data_collator = DataCollatorForLanguageModeling(
# #     tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf"
# # )
