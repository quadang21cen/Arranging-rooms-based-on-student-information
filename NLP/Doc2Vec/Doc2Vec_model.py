# pip install -U kaleido
from underthesea import text_normalize, word_tokenize
from gensim.models import fasttext, word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn import metrics

from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px

import warnings
import time
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

def clean_text(corpus):
    clean_corpus = []
    for i in range(len(corpus)):
        word = str(corpus[i]).lower()
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

def corpus_list(text, stopwords_path):
    corpus = []
    stopwords = get_stopwords_list(stopwords_path)
    for i in range(len(text)):
        word_list = word_tokenize(text[i])
        word_list = [word for word in word_list if not word in stopwords]
        corpus.append(word_list)
    return corpus




def infer_vector_worker(document,model):
    vector = model.infer_vector([document])
    return vector

class Doc2Vec_Class:
    def __init__(self) -> None:
        self.stopwords_path = "NLP\\Doc2Vec\\vietnamese_stopwords.txt"
        self.model = None
    def load(self, path):
        self.model = Doc2Vec.load(path)
    def train(self, df, feature_list, save_folder, vector_size = 100, window = 2, epoch = 100, min_count = 1, min_alpha = 0.0001, alpha = 0.0025):
        start_time = time.time()
        for feature in feature_list:
            clean = clean_text(df[feature])  # chuyen hoa thanh thuong, xoa bo ky tu dac biet
            corpus = corpus_list(clean, stopwords_path=self.stopwords_path)
            tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(corpus)]
            #print(tagged_data)
            # Notice: maintaining a large vector size with tiny data (which allows severe model overfitting)
            # if it's not tens-of-thousands of texts, use a smaller vector size and more epochs (but realize results may still be weak with small data sets)
            # if each text is tiny, use more epochs (but realize results may still be a weaker than with longer texts)
            model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epoch, min_alpha=min_alpha, alpha=alpha)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
            Path(save_folder+"/"+str(feature)).mkdir(parents=True, exist_ok=True)
            model.save(save_folder+"/"+str(feature) + "size {} words {} epochs {}.model".format(vector_size, window, epoch))
            print("FINISH...", feature)
            print("Finish in--- %s seconds ---" % (time.time() - start_time))
    def load_to_matrix(self):
        vectors = self.model.dv.vectors
        return vectors
    def infer_vector_model(self, text, epochs = 50, alpha = 0.25, min_alpha = 0.001):
        vector = self.model.infer_vector(sum(corpus_list(clean_text([text]), stopwords_path=self.stopwords_path),[]), epochs=epochs, alpha=alpha, min_alpha = min_alpha)
        return vector

    def compare_two_unknown_docs(self, text1, text2, min_alpha = 0.0001, alpha = 0.25, epochs = 10):
        return self.model.similarity_unseen_docs(sum(corpus_list(clean_text([text1]), stopwords_path=self.stopwords_path), []),
                                                 sum(corpus_list(clean_text([text2]), stopwords_path=self.stopwords_path), [])
                                                 , min_alpha = min_alpha, alpha = alpha, epochs = epochs)
    def distance(self, text1, text2, min_alpha = 0.0001, alpha = 0.25, epochs = 10):
        vector1 = self.model.infer_vector(sum(corpus_list(clean_text([text1]), stopwords_path=self.stopwords_path), []),
                                          min_alpha = min_alpha, alpha = alpha, epochs = epochs)
        vector2 = self.model.infer_vector(sum(corpus_list(clean_text([text2]), stopwords_path=self.stopwords_path), []),
                                          min_alpha = min_alpha, alpha = alpha, epochs = epochs)
        return cosine_similarity([vector1], [vector2])
    def pairwise_vectors(self, matrix):
        return cosine_similarity(matrix, matrix)
    def pairwise_unknown_docs(self, text_list, min_alpha = 0.0001, alpha = 0.25, epochs = 10):
        # similar_list = []
        # for text1 in text_list:
        #     similar_sublist = []
        #     for text2 in text_list:
        #         sim = self.model.similarity_unseen_docs(
        #             sum(corpus_list(clean_text([text1]), stopwords_path=self.stopwords_path), []),
        #             sum(corpus_list(clean_text([text2]), stopwords_path=self.stopwords_path), [])
        #             , min_alpha=min_alpha, alpha=alpha, epochs=epochs)
        #         similar_sublist.append(sim)
        #     similar_list.append(similar_sublist)
        similar_list = [[self.model.similarity_unseen_docs(
                    sum(corpus_list(clean_text([text1]), stopwords_path=self.stopwords_path), []),
                    sum(corpus_list(clean_text([text2]), stopwords_path=self.stopwords_path), [])
                    , min_alpha=min_alpha, alpha=alpha, epochs=epochs)for text2 in text_list] for text1 in text_list]
        return similar_list

    def cluster_TSNE(self, save_file):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        # doc_tags = model.dv.doctags.keys() # Gensim older version
        doc_tags = self.model.dv.index_to_key

        #print(doc_tags)
        X = model.dv[doc_tags]
        tSNE = TSNE(n_components=2)
        X_tsne = tSNE.fit_transform(X)
        df = pd.DataFrame(X_tsne, index=doc_tags, columns=['x', 'y'])
        plt.figure(0)
        plt.scatter(df['x'], df['y'], s=0.4, alpha=0.4)
        plt.savefig(save_file)

    def birch_score(self, save_file):
        doc_tags = self.model.dv.index_to_key
        X = model.dv[doc_tags]
        k = 3
        brc = Birch(branching_factor=50, n_clusters=k, threshold=0.1, compute_labels=True)
        brc.fit(X)

        clusters = brc.predict(X)
        labels = brc.labels_

        print("Clusters: ")
        print(clusters)

        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

        print("Silhouette_score: ")
        print(silhouette_score)
        with open(save_file, 'w') as f:
            f.write("Silhouette_score: "+str(silhouette_score))

    def find_optimal_clusters(data, max_k):
        iters = range(2, max_k + 1, 2)

        sse = []
        for k in iters:
            sse.append(
                MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
            print('Fit {} clusters'.format(k))
        plt.figure(u'optimal clusters')
        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        plt.show()
    def calculate_inertia(self):
        # calculate k using python, with the elbow method
        inertia = []

        # define our possible k values
        possible_K_values = [i for i in range(2, 40)]

        # we start with 2, as we can not have 0 clusters in k means, and 1 cluster is just a dataset

        # iterate through each of our values
        for each_value in possible_K_values:
            # iterate through, taking each value from
            kmodel = KMeans(n_clusters=each_value, init='k-means++', random_state=32)

            # fit it
            kmodel.fit(self.model.dv.vectors)

            # append the inertia to our array
            inertia.append(kmodel.inertia_)

        plt.plot(possible_K_values, inertia)
        plt.title('The Elbow Method')

        plt.xlabel('Number of Clusters')

        plt.ylabel('Inertia')

        plt.show()
    def calculate_k_for_negatives(self):
        bad_k_values = {}

        # remember, anything past 15 looked really good based on the inertia
        possible_K_values = [i for i in range(15, 30)]

        # we start with 1, as we can not have 0 clusters in k means
        # iterate through each of our values
        for each_value in possible_K_values:

            # iterate through, taking each value from
            model = KMeans(n_clusters=each_value, init='k-means++', random_state=32)

            # fit it
            model.fit(self.model.dv.vectors)

            # find each silhouette score
            silhouette_score_individual = metrics.silhouette_samples(self.model.dv.vectors, model.predict(self.model.dv.vectors))

            # iterate through to find any negative values
            for each_silhouette in silhouette_score_individual:

                # if we find a negative, lets start counting them
                if each_silhouette < 0:

                    if each_value not in bad_k_values:
                        bad_k_values[each_value] = 1

                    else:
                        bad_k_values[each_value] += 1

        for key, val in bad_k_values.items():
            print(f' This Many Clusters: {key} | Number of Negative Values: {val}')
    def semantic_clustering(self,k):
        kmeans_model = KMeans(n_clusters=k)

        kmeans_model.fit(self.model.dv.vectors)
        labels = kmeans_model.labels_
        clusters = kmeans_model.fit_predict(self.model.dv.vectors)

        # Applying PCA to reduce the number of dimensions.
        X = np.array(self.model.dv.vectors)

        pca = PCA(n_components=3)
        result = pca.fit_transform(X)
        centroids = kmeans_model.cluster_centers_
        # create dataframe to feed to

        df = pd.DataFrame({
            'sent': self.model.dv.index_to_key, # Take integer id number from 0
            'cluster': labels.astype(str),
            'x': result[:, 0],
            'y': result[:, 1],
            'z': result[:, 2]
        })
        # Score
        silhouette_score_average  = metrics.silhouette_score(self.model.dv.vectors, kmeans_model.predict(self.model.dv.vectors))
        print("Silhouette_score ( Accuracy Score): ", silhouette_score_average)

        silhouette_score_individual = metrics.silhouette_samples(self.model.dv.vectors, kmeans_model.predict(self.model.dv.vectors))
        negative_score_len = 0
        for each_value in silhouette_score_individual:
            if each_value < 0:
                negative_score_len +=1
        print(f'We have found {negative_score_len} negative silhouette score')
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='cluster', hover_name='sent',
                            range_x=[df.x.min() - 1, df.x.max() + 1],
                            range_y=[df.y.min() - 1, df.y.max() + 1],
                            range_z=[df.z.min() - 1, df.z.max() + 1])
        fig.update_traces(hovertemplate='<b>%{hovertext}</b>')

        fig.show()
        fig.write_image("semantic_clustering.pdf")
    def measure_distance(self, vec_list): # Đây là đo khoảng cách. Trái ngược với đo độ tương đồng similarity

        # compute distance
        distances = (
            metrics.pairwise_distances(
                vec_list[0].reshape(1, -1),
                vec_list[1].reshape(1, -1),
                metric)[0][0] for metric in ["cosine", "manhattan", "euclidean"]
        )
        return distances

    def find_most_similar(self, text, epochs = 50, alpha = 0.0001, topn= 5):
        # try other infer_vector() parameters, such as steps=50 (or more, especially with small texts), and alpha=0.025
        vector = self.model.infer_vector(sum(corpus_list(clean_text([text]), stopwords_path=self.stopwords_path),[]), epochs=epochs, alpha=alpha)
        similar_text = self.model.dv.most_similar(vector, topn=topn)

        return similar_text

if __name__ == '__main__':
    print('Start')
    doc1 = "thích đá bóng và nghe nhạc, xem phim"
    doc2 = "thích thể thao và giải trí"
    doc2vec = Doc2Vec_Class()
    # doc2vec.train(df = file_pd, feature_list = features, save_folder = "model", vector_size = 100, window = 2, epoch = 100)
    doc2vec.load("NLP/Doc2Vec/{}/size {} words {}.model".format("Bio_personality", 20, 5))

    # So sánh 2 đoạn văn bản doc1 và doc2 dùng hàm của gensim doc2vec theo cách tính cosine similarity ( Chuẩn hơn )
    doc_ex1 = "Hướng ngoại, nói chuyện"
    doc_ex2 = "Hướng ngoại 65% và hướng nội 35%, tôi thích nói chuyện với mọi người nhưng đồng thời tôi cũng có những lúc thích một mình"

    # So sánh 2 đoạn văn bản doc1 và doc2 dùng hàm của sklearn.pairwise theo cách tính cosine similarity ( càng thấp thì càng gần giống )
    print(doc2vec.distance(doc_ex1, doc_ex2))

    # # Nếu muốn chuyển text thành vector, dùng hàm này (epochs càng nhiều độ chênh lệch giữa vector của cùng 1 text càng ngắn,
    # # alpha là learning rate)
    # vector = doc2vec.infer_vector_model(doc1, epochs = 50, alpha = 0.025, min_alpha=0.0001)
    # print(vector)

    # # Có thể dùng để recommend ( đề nghị ) các đoạnn văn bản giống nhau nếu dùng hàm này ( topn dùng để recommend bao nhiêu ví dụ gần giống với đoạn văn bản doc)
    # similarity_docs = doc2vec.find_most_similar(doc1, topn=5)
    # similarity_docs_id = [item[0] for item in similarity_docs]
    # print(similarity_docs_id)

    # # Tính similarity dựa vào khoảng cách với vector
    # text_list = ["Document 1", "Document 2"]
    # vectors_for_unknown_text = [doc2vec.infer_vector_model(text) for text in text_list]
    # cosine, manhattan, euclidean = doc2vec.measure_distance(dv_vector)
    # # cosine, manhattan, euclidean = doc2vec.measure_distance(vectors_for_unknown_text)
    # cosine_similar_score = 1 - cosine
    # print("Độ tương đồng của 2 văn bản:", cosine_similar_score)

    # # doc2vec.calculate_inertia()

    # # Clustering các đoạn văn bản dùng Kmeans và PCA
    # # Giá trị silhouette từ [-1; 1] : Số 1 là tốt nhất, -1 là tệ nhất, 0 là trùng nhau
    # doc2vec.semantic_clustering(k=49)