from sklearn.metrics.pairwise import cosine_similarity
from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

class TF_IDF_class:
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
        vect = TfidfVectorizer(tokenizer=self.tokenize_vn, stop_words=stop_words_list, lowercase=True, min_df=0)
        tfidf = vect.fit_transform(text_list)
        return vect,tfidf
    def text2vec(self, text_list):
        vect, tfidf = self.transform_vector(text_list)
        return tfidf.toarray()

import pandas as pd
df = pd.read_csv('Student_Ins.csv')
features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                             "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
df.columns = features

corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
            "Toi thich da bong"
           ]      
# stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
# vect = TfidfVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase = True)
# tfidf = vect.fit_transform(corpus)

tf_idf = TF_IDF_class()
vect, tfidf = tf_idf.transform_vector(df['Bio_personality'])

import pandas as pd
feature_df = pd.DataFrame(tf_idf.text2vec(df['Bio_personality']),
                        columns=vect.get_feature_names_out())
#print(tfidf.shape)
print(df['Bio_personality'].shape)
print(tfidf.toarray())
arrays = [value for value in tfidf.toarray()]
print(tfidf.toarray())
print(arrays)
df['Bio_personality'] = arrays
print(df['Bio_personality'])

feature_df.to_csv("test.csv")

pairwise_similarity = tfidf * tfidf.T 

print("Do tuong dong:",pairwise_similarity)


from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

class TF_IDF_class:
    def __init__(self) -> None:
        self.stopwords_path = "NLP\\TF_IDF\\vietnamese_stopwords.txt"

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
        vect = TfidfVectorizer(tokenizer=self.tokenize_vn, stop_words=stop_words_list, lowercase=True, min_df=0)
        tfidf = vect.fit_transform(text_list)
        return vect,tfidf
    def text2vec(self, text_list):
        vect, tfidf = self.transform_vector(text_list)
        arrays = [value for value in tfidf.toarray()]
        return arrays
    def pairwise(self, matrix):
        return cosine_similarity(matrix, matrix)
    def compare_vectors(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])



def test():
    print("AAA")

import pandas as pd
df = pd.read_csv('Student_Ins.csv')
features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                             "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
df.columns = features


corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
            "Toi thich da bong"
           ]      
# stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
# vect = TfidfVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase = True)
# tfidf = vect.fit_transform(corpus)

tf_idf = TF_IDF_class()
vect, tfidf = tf_idf.transform_vector(df['Bio_personality'])

import pandas as pd
feature_df = pd.DataFrame(tfidf.todense(),
                        columns=vect.get_feature_names_out())

feature_df.to_csv("test.csv")

pairwise_similarity = tfidf * tfidf.T 
matrix = tf_idf.text2vec(df['Bio_personality'])
print(tf_idf.pairwise(matrix))
cosine_similarity_pd = pd.DataFrame(tf_idf.pairwise(matrix), columns = [*range(len(matrix))])
print(cosine_similarity_pd)