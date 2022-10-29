from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re


class Bag_Of_Word:
    def __init__(self) -> None:
<<<<<<< HEAD
        self.stopwords_path = "NLP\\Bag Of Words\\vietnamese_stopwords.txt"
=======
        self.stopwords_path = "vietnamese_stopwords.txt"
>>>>>>> 022a1314baa1846d8975896f0a6d4a13db48a56b

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
<<<<<<< HEAD
        arrays = [value for value in BOW.toarray()]
        return vect.get_feature_names_out(), arrays
    
=======
        return BOW.toarray()

    def distance(self, matrix):
        return cosine_similarity(matrix, matrix)

>>>>>>> 022a1314baa1846d8975896f0a6d4a13db48a56b
if __name__ == '__main__':
    corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
<<<<<<< HEAD
              "Tao thich nhay mua"]
    # stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
    # vect = CountVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase=True)
    # BOW = vect.fit_transform(corpus)
    bow = Bag_Of_Word()
    librarey, bow.text2vec(corpus)
    print(bow.text2vec(corpus))

    # vect, matrix = bow.transform_vector(corpus)
=======
              "Tao thich nhay mua"
              ]
    import pandas as pd
    file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
    print(file_pd.columns)
    features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                                         "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
    file_pd.columns = features
    bow = Bag_Of_Word()
    matrix = bow.text2vec(file_pd["hobby_interests"])
>>>>>>> 022a1314baa1846d8975896f0a6d4a13db48a56b

    # import pandas as pd

<<<<<<< HEAD
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
=======

    print("Cosine similarity")
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity = cosine_similarity(matrix, matrix)

    cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(matrix.toarray()))])
    print(cosine_similarity_pd)

    print(bow.distance(matrix))
>>>>>>> 022a1314baa1846d8975896f0a6d4a13db48a56b
