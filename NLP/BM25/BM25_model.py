
import pandas as pd
from rank_bm25 import *
from underthesea import text_normalize, word_tokenize
import re
import string
class BM25_class:
    def __init__(self) -> None:
        self.stopwords_path = "NLP\\BM25\\vietnamese_stopwords.txt"
    def get_stopwords_list(self, stop_file_path):
        """load stop words """

        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))
    def tokenize_vn(self, doc):
        # Tokenize the words
        doc = doc.lower()
        normalized_doc = text_normalize(doc)
        doc = word_tokenize(normalized_doc)
        # Remove Punctuation (VD: Xoa dau !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ )
        punctuation = string.punctuation
        clean_words = [w for w in doc if w not in punctuation]
        clean_words = [w for w in clean_words if (" " in w) or w.isalpha()]
        stop_words_list = self.get_stopwords_list(self.stopwords_path)
        clean_words = [word for word in clean_words if not word in stop_words_list]
        return clean_words
    def tokenize_vn_docs(self, docs):
        # Tokenize the words
        corpus = []
        for doc in docs:
            clean_words = self.tokenize_vn(doc)
            corpus.append(clean_words)

        return corpus
    def transform_vector(self, text_list):
        # Create TfidfVectorizer for Vietnamese
        text_corpus = self.tokenize_vn_docs(text_list)
        vect = BM25Okapi(text_corpus)
        return vect
    def pairwise(self, vec, text_list):
        corpus = self.tokenize_vn_docs(text_list)
        sim_matrix = []
        for words in corpus:
            sim = vec.get_scores(words)
            sim_matrix.append(sim)
        return sim_matrix

if __name__ == '__main__':
    # Giống như doc2vec (search google gợi ý câu gần nhất)
    corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
              "Tao thich nhay mua"
              ]

    BM25_instance = BM25_class()
    vec = BM25_instance.transform_vector(corpus)
    query1 = "Ban dang boi loi, nghe nhac"
    tokenized_query = BM25_instance.tokenize_vn(query)

    # Cho ra 1 array chứa giá trị so sánh giữa query và corpus
    print(vec.get_scores(tokenized_query))

    # Cho ra câu gần nhất với query
    print(vec.get_top_n(tokenized_query, corpus, n=1))

    # pairwise matrix cho BM25
    sim_matrix = BM25_instance.pairwise(vec,corpus)
    sim_matrix_pd = pd.DataFrame(sim_matrix, columns=[*range(len(sim_matrix))])

import pandas as pd
from rank_bm25 import *
from underthesea import text_normalize, word_tokenize
import re
import string
class BM25_class:
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
        doc = doc.lower()
        normalized_doc = text_normalize(doc)
        doc = word_tokenize(normalized_doc)
        # Remove Punctuation (VD: Xoa dau !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ )
        punctuation = string.punctuation
        clean_words = [w for w in doc if w not in punctuation]
        clean_words = [w for w in clean_words if (" " in w) or w.isalpha()]
        stop_words_list = self.get_stopwords_list(self.stopwords_path)
        clean_words = [word for word in clean_words if not word in stop_words_list]
        return clean_words
    def tokenize_vn_docs(self, docs):
        # Tokenize the words
        corpus = []
        for doc in docs:
            clean_words = self.tokenize_vn(doc)
            corpus.append(clean_words)

        return corpus
    def transform_vector(self, text_list):
        # Create TfidfVectorizer for Vietnamese
        text_corpus = self.tokenize_vn_docs(text_list)
        vect = BM25Okapi(text_corpus)
        return vect
    def pairwise(self, vec, text_list):
        corpus = self.tokenize_vn_docs(text_list)
        sim_matrix = []
        for words in corpus:
            sim = vec.get_scores(words)
            sim_matrix.append(sim)
        return sim_matrix

if __name__ == '__main__':
    # Giống như doc2vec (search google gợi ý câu gần nhất)
    corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
              "Tao thich nhay mua"
              ]

    BM25_instance = BM25_class()
    vec = BM25_instance.transform_vector(corpus)
    query = "Ban dang boi loi, nghe nhac"
    tokenized_query = BM25_instance.tokenize_vn(query)

    # Cho ra 1 array chứa giá trị so sánh giữa query và corpus
    print(vec.get_scores(tokenized_query))

    # Cho ra câu gần nhất với query
    print(vec.get_top_n(tokenized_query, corpus, n=1))

    # pairwise matrix cho BM25
    sim_matrix = BM25_instance.pairwise(vec,corpus)
    sim_matrix_pd = pd.DataFrame(sim_matrix, columns=[*range(len(sim_matrix))])
    print(sim_matrix_pd)