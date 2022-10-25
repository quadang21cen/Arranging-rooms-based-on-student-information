from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
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
            "Toi thich da bong"
           ]      
stop_words_list = get_stopwords_list(".\\vietnamese_stopwords.txt")
vect = TfidfVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list, lowercase = True)
tfidf = vect.fit_transform(corpus)

import pandas as pd
feature_df = pd.DataFrame(tfidf.toarray(),
                        columns=vect.get_feature_names_out())
print(feature_df)

pairwise_similarity = tfidf * tfidf.T 

print("Do tuong dong:",pairwise_similarity)