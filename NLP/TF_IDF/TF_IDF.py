from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string

class TF_IDF:
    def __init__(self) -> None:
        self.stopwords_path = ".\\vietnamese_stopwords.txt"

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
        vect = TfidfVectorizer(tokenizer=self.tokenize_vn, stop_words=stop_words_list, lowercase=True)
        tfidf = vect.fit_transform(text_list)
        return vect,tfidf
    def text2vec(self, text_list):
        vect, tfidf = self.transform_vector(text_list)
        return tfidf.toarray()

def test():
    print("AAA")

corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
            "Toi thich da bong"
           ]      
# stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
# vect = TfidfVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase = True)
# tfidf = vect.fit_transform(corpus)

tf_idf = TF_IDF()
vect, tfidf = tf_idf.transform_vector(corpus)

import pandas as pd
feature_df = pd.DataFrame(tf_idf.text2vec(corpus),
                        columns=vect.get_feature_names_out())
print(feature_df)

pairwise_similarity = tfidf * tfidf.T 

print("Do tuong dong:",pairwise_similarity)

