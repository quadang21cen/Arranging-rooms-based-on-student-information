from underthesea import sent_tokenize, text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re


def get_stopwords_list(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))


def tokenize_vn(doc):
    # Remove Numbers (Dung de xoa so)
    # doc = re.sub(r'\d+', '', doc)

    punctuation = string.punctuation
    normalized_doc = text_normalize(doc)
    doc = word_tokenize(normalized_doc)
    # Remove Punctuation (VD: Xoa dau !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ )
    clean_words = [w for w in doc if w not in punctuation]
    return clean_words


corpus = ["tôi  thích bơi lội,nghe nhạc, và đọc sách",
          "Toi thich da bong",
          "Toi thich boi loi",
          "Ban dang boi loi, nghe nhac",
          "Tao thich nhay mua"
          ]
stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
vect = CountVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase=True)
BOW = vect.fit_transform(corpus)

import pandas as pd

feature_df = pd.DataFrame(BOW.toarray(),
                          columns=vect.get_feature_names_out())

from scipy import spatial
print(BOW.toarray()[0],len(BOW.toarray()[0]))
print(BOW.toarray()[1],len(BOW.toarray()[1]))
# print("(0,0):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[0]))
# print("(0,1):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[1]))
# print("(0,2):",spatial.distance.cosine(BOW.toarray()[0], BOW.toarray()[2]))
# print("(1,0):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[0]))
# print("(1,1):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[1]))
# print("(1,2):",spatial.distance.cosine(BOW.toarray()[1], BOW.toarray()[2]))
# print("(2,2):",spatial.distance.cosine(BOW.toarray()[2], BOW.toarray()[2]))

print("Cosine similarity")
from sklearn.metrics.pairwise import linear_kernel

cosine_similarity = linear_kernel(BOW, BOW)

cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(BOW.toarray()))])
print(cosine_similarity_pd)
