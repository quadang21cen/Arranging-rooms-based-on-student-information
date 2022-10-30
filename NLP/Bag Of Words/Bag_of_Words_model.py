from underthesea import text_normalize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
class Bag_Of_Word:
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
        clean_words = [w for w in clean_words if (" " in w) or w.isalpha()]
        return clean_words

    def transform_vector(self, text_list):
        # Create TfidfVectorizer for Vietnamese
        stop_words_list = self.get_stopwords_list(self.stopwords_path)
        vect = CountVectorizer(tokenizer=self.tokenize_vn, stop_words=stop_words_list, lowercase=True, ngram_range=(1,3), max_features=3000)
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

    import random

    file_pd["labels"] = [random.randint(0, 5) for _ in range(len(file_pd['hobby_interests']))]

    bow = Bag_Of_Word()
    matrix = bow.text2vec(file_pd["hobby_interests"])

    # so sánh 2 vector
    cosine = bow.compare_vectors(matrix[0], matrix[1])
    print(cosine)
    import pandas as pd

    print("Cosine similarity")

    cosine_similarity = bow.pairwise(matrix)
    print(cosine_similarity)
    cosine_similarity_pd = pd.DataFrame(cosine_similarity, columns = [*range(len(matrix))])
    print(cosine_similarity_pd)

    # Đánh giá độ chính xác bằng text classification

    from matplotlib import pyplot as plt
    def plot():
        fig = plt.figure(figsize=(10, 10))
        file_pd.groupby('labels').Bio_personality.count().sort_values().plot.barh(ylim=0,
                                                                             title='Number of records per label\n')
        plt.xlabel('Number of records.')
        # plt.show()


    plot()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from xgboost import XGBClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score

    models = [
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        XGBClassifier(),
        LinearSVC(),
        MultinomialNB()
    ]

    labels = file_pd.labels

    cross_value_scored = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, matrix, labels, scoring='accuracy', cv=5)
        for accuracy in accuracies:
            cross_value_scored.append((model_name, accuracy))

    df_cv = pd.DataFrame(cross_value_scored, columns=['model_name', 'accuracy'])
    acc = pd.concat([df_cv.groupby('model_name').accuracy.mean(), df_cv.groupby('model_name').accuracy.std()], axis=1,
                    ignore_index=True)
    acc.columns = ['Mean Accuracy', 'Standard deviation']
    print(acc)

    # hyperparameter tunning of XGBClassifier
    clf_xgb = XGBClassifier()
    # hypermeter setting
    param_dist = {'n_estimators': np.random.randint(150, 500, 100),
                  'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.59],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'min_child_weight': [1, 2, 3, 4]
                  }
    from sklearn.model_selection import KFold

    kfold_5 = KFold(shuffle=True, n_splits=5)

    from sklearn.model_selection import RandomizedSearchCV

    clf = RandomizedSearchCV(clf_xgb,
                             param_distributions=param_dist,
                             cv=kfold_5,
                             n_iter=5,
                             scoring='roc_auc',
                             verbose=3,
                             n_jobs=-1)

    # spliting the data into test and train
    x_train, x_test, y_train, y_test = train_test_split(matrix, labels, random_state=0, test_size=0.2)

    # creating and fiting the instance of the xgb classifier
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1,
                  objective='multi:softprob', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)

    y_pred = xgb.predict(x_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print('Classification report')
    print(classification_report(y_test, y_pred))

    # confusion_matrx
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots(figsize=(11,11))
    # label_id_df = df[['labels', "Bio_personality"]].drop_duplicates()
    # sns.heatmap(conf_matrix, annot=True, cmap = 'Reds', fmt='d',xticklabels = label_id_df.labels.values,
    #            yticklabels = label_id_df.labels.values,square = True )
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.title('Confusion Matrix.')
    plt.show()

    print('Accuracy of the model {}'.format(accuracy_score(y_test, y_pred)))