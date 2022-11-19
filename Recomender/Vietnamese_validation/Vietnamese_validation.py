from typing import List, Iterable, Generator
import itertools
import os.path
import re
import underthesea

class VietTrie:
  def __init__(self) -> None:
    self.next = {}
    self.is_word = False

  def has_word(self, word: str) -> bool:
    tokens = word.split(" ")
    tmp = self
    for token in tokens:
      if token not in tmp.next:
        return False
      tmp = tmp.next[token]

    return tmp.is_word


  def add_word(self, word: str) -> None:
    tokens = word.lower().split(" ")
    tmp = self
    for token in tokens:
      if token not in tmp.next:
        tmp.next[token] = self.__class__() # a hack to make VietTrie singleton :)
      tmp = tmp.next[token]
    tmp.is_word = True

words = []
with open(os.path.join(os.path.dirname(__file__), "words.txt"), "r", encoding="utf8") as f:
  words = f.read().split("\n")
# a hack to make VietTrie singleton :)
VietTrie = VietTrie()

for word in words:
  VietTrie.add_word(word)


def isMeaning(text):
    list_tokens = underthesea.word_tokenize(text)
    words = []
    num_not_mean = 0
    for token in list_tokens:
        if VietTrie.has_word(token.lower()):
            words.append(token)
        else:
            num_not_mean = num_not_mean + 1
    if num_not_mean/len(list_tokens) < 0.7:
        return True
    return False


if __name__ == "__main__":

  sentence = 'Đồng Trống Asgard Hạ Nội Con con'
  print(isMeaning(sentence))










