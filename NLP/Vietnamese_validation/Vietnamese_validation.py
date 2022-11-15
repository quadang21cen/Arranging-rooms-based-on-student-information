from viet_trie import VietTrie
import underthesea

sentence = 'Đồng Trống ABCDFG FGTGWAGSF Asgard Hạ Nội Con con'
def split_words(text):
    list_tokens = underthesea.word_tokenize(text)
    words = []
    for token in list_tokens:
        if VietTrie.has_word(token.lower()):
            print(token)
            words.append(token)

print(split_words(sentence))