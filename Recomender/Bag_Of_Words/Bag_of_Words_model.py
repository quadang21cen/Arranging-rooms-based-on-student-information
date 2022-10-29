from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import warnings

warnings.filterwarnings("ignore")
class Bag_Of_Word:
    def __init__(self) -> None:
        self.stopwords_path = "NLP\\Bag Of Words\\vietnamese_stopwords.txt"

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
        arrays = [value for value in BOW.toarray()]
        return arrays
    
if __name__ == '__main__':
    corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
              "Tao thich nhay mua"]
    # stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
    # vect = CountVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase=True)
    # BOW = vect.fit_transform(corpus)
    bow = Bag_Of_Word()
    bow.text2vec(corpus)
    print(bow.text2vec(corpus))

    # vect, matrix = bow.transform_vector(corpus)

    # import pandas as pd

    # feature_df = pd.DataFrame(bow.text2vec(corpus),
    #                           columns=vect.get_feature_names_out())

    # from scipy import spatial
    # print(matrix.toarray()[0],len(matrix.toarray()[0]))
    # print(matrix.toarray()[1],len(matrix.toarray()[1]))
    # # print("(0,0):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[0]))
    # # print("(0,1):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[1]))
    # # print("(0,2):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[2]))
    # # print("(1,0):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[0]))
    # # print("(1,1):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[1]))
    # # print("(1,2):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[2]))
    # # print("(2,2):",spatial.distance.cosine(BOW.toarray()[2], BOW.toarray()[2]))

    # print("Cosine similarity")
    # from sklearn.metrics.pairwise import linear_kernel

    # cosine_similarity = linear_kernel(matrix, matrix)

    # cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(matrix.toarray()))])
    # print(cosine_similarity_pd)
