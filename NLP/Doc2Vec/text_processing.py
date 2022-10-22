from underthesea import sent_tokenize, text_normalize, word_tokenize
from gensim.models import fasttext, word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from nltk.corpus import stopwords

import numpy as np

import warnings
warnings.filterwarnings('ignore')

def clean_text(corpus):
    clean_corpus = []
    for i in range(len(corpus)):
        word = corpus[i].lower()
        word = text_normalize(word)
        word = re.sub(r"\s+", " ", word) # Remove multiple spaces in content
        # remove punctuation
        #word = re.sub('[^a-zA-Z]', ' ', word)

        # remove digits and special chars
        word = re.sub("(\\d|\\W)+", " ", word)
        clean_corpus.append(word)
    return clean_corpus


def get_stopwords_list(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

def corpus_list(text):
    corpus = []
    stopwords_path = "Continuous_BOW and Doc2Vec\\vietnamese_stopwords.txt"
    stopwords = get_stopwords_list(stopwords_path)
    for i in range(len(text)):
        word_list = word_tokenize(text[i])
        word_list = [word for word in word_list if not word in stopwords]
        #for n in range(len(word_list)):
        #    word_list[n] = re.sub(" ", "_", word_list[n])
        corpus.append(word_list)
    return corpus




def infer_vector_worker(document,model):
    vector = model.infer_vector([document])
    return vector



if __name__ == '__main__':
    hobit_A =  ' Minh # la mot nguoi huong noi thich boi loi va doc sach,thich an mi cay, uong caffe'
    hobit_B = 'minh la mot nguoi thich da bong va nghe nhac'
    # hobit = hobit.split(",")
    # clean = clean_text(hobit) # chuyen hoa thanh thuong, xoa bo ky tu dac biet
    # corpus = corpus_list(clean)
    # tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(corpus)]
    # print(tagged_data)
    # model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    # model.save("Testing.model")
    # print("FINISH...")

    model = Doc2Vec.load("Testing.model")
    vector_A = model.infer_vector(corpus_list(clean_text([hobit_A])))
    
    vector_B = model.infer_vector(corpus_list(clean_text([hobit_B])))
    # Assess the model
    test_data = corpus_list(clean_text(["minh la ai"]))
    inferred_vector = model.infer_vector(test_data)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tagged_data[sims[index][0]].words)))

    # Compare 2 unknown docs
    print("VEC_A:", vector_A,len(vector_A))
    print("VEC_B:", vector_B,len(vector_B))

    from sklearn.metrics.pairwise import linear_kernel

    cosine_similarity = linear_kernel(vector_A, vector_B)
    print(cosine_similarity)
    
    

    # # Đây là nơi gensim biến đổi text thành vector
    # test_doc = word_tokenize("đá bóng".lower())
    # test_doc_vector = model.infer_vector(test_doc)
    # print("Text to vec:", test_doc_vector)
    # print("Độ tương đồng:", model.docvecs.most_similar(positive=test_doc_vector))

    # print(model.wv.most_similar("bóng đá")[:5])
    # print(sum(corpus,[]))
    # vectors = model.infer_vector(["mình là một người hướng nội, ít nói chuyện thích ăn mì cay, uống caffee vào mỗi buổi sáng thích khoa học, thích đá bóng, bơi lội."])
    # print(vectors)
    # from sklearn.decomposition import PCA

    # from matplotlib import pyplot as plt

    # X = model[model.wv.key_to_index.keys()]
    # pca = PCA(n_components=2)
    # plt.figure(1)
    # result = pca.fit_transform(X)

    # # create a scatter plot of the projection
    # plt.scatter(result[:, 0], result[:, 1])
    # words = list(model.wv.key_to_index.keys())

    # for i, word in enumerate(words):
    #     plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    # from scipy import spatial

    # vec1 = model.infer_vector(corpus[0])
    # vec2 = model.infer_vector(corpus[1])

    # cos_distance = spatial.distance.cosine(vec1, vec2)

    # print(cos_distance)

    # # Train Continuous BOW
    # from Continuous_BOW import gradient_descent, get_dict, compute_pca

    # C = 2
    # N = 10
    # word2Ind, Ind2word = get_dict(sum(corpus,[]))
    # V = len(word2Ind)
    # num_iters = 150
    # print("Call gradient_descent")
    # W1, W2, b1, b2 = gradient_descent(sum(corpus,[]), word2Ind, N, V, num_iters)
    # word_saved = sum(corpus,[])

    # embs = (W1.T + W2) / 2.0

    # idx = [word2Ind[word] for word in sum(corpus,[])]
    # dict_for_word = dict(zip(sum(corpus,[]), idx))   # Save dict_for_word
    # print(word2Ind)
    # X = embs[idx, :]
    # print(embs[dict_for_word.get("bóng đá")])
    # print(pd.DataFrame(embs[idx, :], index=sum(corpus,[])).head())
    # #print(X.shape, idx)
    # #print(X)


    # plt.figure(2)
    # result = compute_pca(X, 2)
    # plt.scatter(result[:, 0], result[:, 1])
    # for i, word in enumerate(sum(corpus,[])):
    #     plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    # plt.show()

    # import numpy as np
    # ixd_vec3 = [word2Ind[word] for word in sum(corpus_list(clean_text(["mình là một người hướng nội, ít nói chuyện thích ăn mì cay, uống caffee vào mỗi buổi sáng thích khoa học, thích đá bóng, bơi lội."])), [])]
    # ixd_vec4 = [word2Ind[word] for word in corpus[1]]
    # print(corpus[0])
    # print("Vector 1:",np.array(embs[ixd_vec3]).mean(axis=0))
    # print(corpus[1])
    # print("Vector 2:",np.array(embs[ixd_vec4]).mean(axis=0))

    # cos_distance1 = spatial.distance.cosine(np.array(embs[ixd_vec3]).mean(axis=0), np.array(embs[ixd_vec4]).mean(axis=0))
    # print(cos_distance1)
