from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from sklearn.metrics.pairwise import cosine_similarity

class Bag_Of_Word:
    def __init__(self) -> None:
        self.stopwords_path = "NLP\\Bag_Of_Words\\vietnamese_stopwords.txt"

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

    def pairwise(self, matrix):
        return cosine_similarity(matrix, matrix)

    def compare_vectors(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])

if __name__ == '__main__':
    corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
              "Toi thich da bong",
              "Toi thich boi loi",
              "Ban dang boi loi, nghe nhac",
              "Tao thich nhay mua"
              ]
    import pandas as pd
    file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
    #print(file_pd.columns)
    features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                                         "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
    file_pd.columns = features
    bow = Bag_Of_Word()
    matrix = bow.text2vec(file_pd["hobby_interests"])

    # so sánh 2 vector
    cosine = bow.compare_vectors(matrix[0], matrix[1])
    print(cosine)
    import pandas as pd


    print("Cosine similarity")

    cosine_similarity = bow.pairwise(matrix)

    cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(matrix))])
    print(cosine_similarity_pd)

    print(bow.distance(matrix))