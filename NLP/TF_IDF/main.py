from underthesea import sent_tokenize, text_normalize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def get_stopwords_list(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))
def tokenize_vn(doc):
    normalized_doc = text_normalize(doc)
    doc = word_tokenize(normalized_doc)
    return doc




corpus = ["tôi thích bơi lội,nghe nhạc, và đọc sách", 
           ""
           ]      
stop_words_list = get_stopwords_list("C:\\Users\\quach\\Desktop\\Recent\\RS_demo\\vietnamese_stopwords.txt")                                                                                                                                                                                             
vect = TfidfVectorizer(tokenizer=tokenize_vn, stop_words=stop_words_list)                                                                                                                                                                                                   
tfidf = vect.fit_transform(corpus)                                                                                                                                                                                                                       
pairwise_similarity = tfidf * tfidf.T 

print(pairwise_similarity)