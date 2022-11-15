from viet_trie import VietTrie
import underthesea

sentence = 'Đồng Trống ABCDFG FGTGWAGSF Asgard Hạ Nội Con con'
list_tokens = underthesea.word_tokenize(sentence)
print(list_tokens)

words = []
for token in list_tokens:
    if VietTrie.has_word(token.lower()):
        print(token)
        words.append(token)
print(words)