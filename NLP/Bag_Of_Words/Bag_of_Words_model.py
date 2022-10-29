from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re


class Bag_Of_Word:
    def __init__(self) -> None:
        self.stopwords_path = "vietnamese_stopwords.txt"

    def get_stopwords_list(self, stop_file_path):
        """load stop words """

        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))

    def tokenize_vn(self, doc):
        # Tokenize the words
        normalized_doc = text_normalize(doc)
        doc = word_tokenize(normalized_doc)
        # Remove Punctuation (VD: Xoa dau !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ )
        punctuation = string.punctuation
        clean_words = [w for w in doc if w not in punctuation]
        return clean_words
    def transform_vector(self, text_list):
        # Create TfidfVectorizer for Vietnamese
        stop_words_list = self.get_stopwords_list(self.stopwords_path)
        vect = CountVectorizer(tokenizer=self.tokenize_vn, stop_words=stop_words_list, lowercase=True)
        tfidf = vect.fit_transform(text_list)
        return vect,tfidf
    def text2vec(self, text_list):
        vect, BOW = self.transform_vector(text_list)
        return BOW.toarray()

if __name__ == '__main__':
    corpus = ["toi thich choi da bong va an uong","toi thich choi da bong"]
    bow = Bag_Of_Word()
    a = bow.text2vec(corpus)
    print(a)


