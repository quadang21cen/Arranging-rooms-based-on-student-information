from viet_trie import VietTrie
import underthesea
import re
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
def isMeaning(text):
    text = standardize_data(text)
    list_tokens = underthesea.word_tokenize(text)
    words = []
    num_not_mean = 0
    for token in list_tokens:
        if VietTrie.has_word(token.lower()):
            words.append(token)
        else:
            num_not_mean = num_not_mean + 1
    if num_not_mean/len(list_tokens) < 0.6:
        return True
    return False
if __name__ == "__main__":
    sentence = 'Đồng Trống Asgard Hạ Nội Con con'
    print(isMeaning(sentence))