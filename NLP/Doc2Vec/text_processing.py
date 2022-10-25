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
    stopwords_path = "./vietnamese_stopwords.txt"
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
    file_pd = pd.read_csv("Student_Ins.csv", encoding='utf-8')
    features = ["Timestamp", "Name", "Sex", "Hometown", "Major", "Bio_personality", "food_drink", "hobby_interests",
                "smoking", "refer_roommate", "Cleanliess", "Privacy", "Unnamed"]
    file_pd.columns = features
    file_pd = file_pd.drop(columns=["Timestamp", "Unnamed"], axis=1)
    hobit_A =  'Mì cay'
    hobit_B = 'bánh mì'
    # hobit = [' Minh # la mot nguoi huong noi thich boi loi va doc sach,thich an mi cay, uong caffe', 'minh la mot nguoi thich da bong va nghe nhac']
    list_features = ["Bio_personality", "food_drink", "hobby_interests"]
    # Dimensionality of the feature vectors.
    # Typically as Word2Vec, more dimensions = greater quality encoding, but there will be some limit beyond which you'll get diminishing returns.
    size = 100
    # The maximum distance between the current and predicted word within a sentence.
    # Notice: the paragraph-vectors approach already has a whole-document window-size so
    # the 'window' parameter just affects how many nearby words are also mixed-in (with the doc-vector) to help predict one target word
    words_surrounding_term = 10
    # for feature in list_features:
    #     clean = clean_text(file_pd[feature]) # chuyen hoa thanh thuong, xoa bo ky tu dac biet
    #     corpus = corpus_list(clean)
    #     tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(corpus)]
    #     print(tagged_data)
    #     model = Doc2Vec(vector_size=size, window=words_surrounding_term, min_count=1, epochs=100)
    #     model.build_vocab(tagged_data)
    #     model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    #     model.save("./{}/size {} words {}.model".format(feature, size, words_surrounding_term))
    #     print("FINISH...", feature)

    print("-------------------------------------------------------")
    model = Doc2Vec.load("./{}/size {} words {}.model".format(list_features[1], size, words_surrounding_term))
    tokenized_hobit_A = sum(corpus_list(clean_text(hobit_A)), [])
    vector_A = model.infer_vector(tokenized_hobit_A,epochs= 100)
    tokenized_hobit_B = sum(corpus_list(clean_text(hobit_B)), [])
    vector_B = model.infer_vector(tokenized_hobit_B,epochs= 100)
    print(hobit_A)
    print(hobit_B)
    print("VEC_A:", vector_A,len(vector_A))
    print("VEC_B:", vector_B,len(vector_B))

    from scipy import spatial

    # cos_distance indicates how much the two texts differ from each other:
    # higher values mean more distant(i.e.different) texts
    # Notice: All of them are expecting TaggedDocument
    # The range is -1 to 1 but there is some rounding error, so I ended up making the values -1 if they were less that -1 ans same for 1
    # https://stackoverflow.com/questions/53503049/measure-similarity-between-two-documents-using-doc2vec
    cos_distance = spatial.distance.cosine(vector_A, vector_B)
    print(cos_distance)
    # Find similarity in Doc2Vec model
    similar_doc = model.dv.most_similar(vector_A, topn=5)
    print("Similar Docs to the vector:")
    print(similar_doc)
    # test_data = corpus_list(clean_text(file_pd[list_features[1]]))
    # tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(test_data)]
    # for i in range(5):
    # print(' '.join(tagged_data[sims[index][0]].words))
    print("-----------------------------------")

    # Evaluate the Model
    test_data = corpus_list(clean_text(file_pd[list_features[1]]))
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(test_data)]
    ranks = []
    second_ranks = []
    for doc_id in range(len(tagged_data)):
        inferred_vector = model.infer_vector(tagged_data[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    import collections

    counter = collections.Counter(ranks)
    print(counter)


    import random
    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_data) - 1)
    inferred_vector = model.infer_vector(test_data[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print("-------------------------------------")
    print('\nTest Document ({}): «{}»'.format(doc_id, ' '.join(test_data[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tagged_data[sims[index][0]].words)))
    X = []
    start_alpha = 0.01
    infer_epoch = 1000

    for d in test_data:
        X.append(model.infer_vector(d, alpha=start_alpha, epochs=infer_epoch))
    k = 3

    from sklearn.cluster import Birch
    from sklearn import metrics

    brc = Birch(branching_factor=50, n_clusters=k, threshold=0.1, compute_labels=True)
    brc.fit(X)

    clusters = brc.predict(X)
    values = [' '.join(words) for words in test_data]
    labels = brc.labels_

    print("Clusters: ")
    print(clusters)

    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

    print("Silhouette_score: ")
    print(silhouette_score)
    from matplotlib import pyplot as plt
    from matplotlib import cm
    plt.figure(num=0, figsize=(18, 11), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter([*range(len(values))], labels, c=clusters)
    plt.savefig("./{}/size {} words {}.png".format(list_features[1], 50, 5))


    model = Doc2Vec.load("./{}/size {} words {}.model".format(list_features[1], 20, 2))

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_data) - 1)
    inferred_vector = model.infer_vector(test_data[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print("-------------------------------------")
    print('Test Document ({}): «{}»'.format(doc_id, ' '.join(test_data[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tagged_data[sims[index][0]].words)))
    with open("./{}/size {} words {}.txt".format(list_features[1], 20, 2), 'w', encoding='utf-8') as f:
        f.write('\nTest Document '+ str(doc_id)+" : "+ ' '.join(test_data[doc_id]))
        f.write(u'\nSIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            f.write("\n%s"% label +" "+"("+str(sims[index][0])+", "+str(sims[index][1]) +") : "+ ' '.join(tagged_data[sims[index][0]].words))
    X = []
    start_alpha = 0.01
    infer_epoch = 1000

    for d in test_data:
        X.append(model.infer_vector(d, alpha=start_alpha, epochs=infer_epoch))
    k = 3

    from sklearn.cluster import Birch
    from sklearn import metrics

    brc = Birch(branching_factor=50, n_clusters=k, threshold=0.1, compute_labels=True)
    brc.fit(X)

    clusters = brc.predict(X)
    values = [' '.join(words) for words in test_data]
    labels = brc.labels_
    print(len(values))
    print(len(labels))

    print("Clusters: ")
    print(clusters)

    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    print(len(values))
    print(len(labels))

    print("Silhouette_score: ")
    print(silhouette_score)
    with open("./{}/size {} words {} Silhouette_score.txt".format(list_features[1], 20, 2), 'w') as f:
        f.write("Silhouette_score: "+str(silhouette_score))

    from sklearn.decomposition import PCA
    X = model[list(model.wv.index_to_key)]
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    colors = np.arange(len(X_pca[:,0]))
    fig = plt.figure(num=1, figsize=(18, 11), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c= colors, cmap="Blues", s=60)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.savefig("./{}/size {} words {}.pdf".format(list_features[1], 20, 2))

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
