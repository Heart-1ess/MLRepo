from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import numpy as np
import math

from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.svm import SVC
from textblob import TextBlob

# from exam import lda

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import math
from wordcloud import WordCloud, STOPWORDS
from sklearn.svm import SVC
from textblob import TextBlob
from PIL import Image
# 将聚类结果标签转换为原始文本数据
def labels_to_original(labels, forclusterlist):
    assert len(labels) == len(forclusterlist)
    maxlabel = max(labels)
    numberlabel = [i for i in range(0, maxlabel + 1, 1)]
    numberlabel.append(-1)
    result = [[] for i in range(len(numberlabel))]
    for i in range(len(labels)):
        index = numberlabel.index(labels[i])
        result[index].append(forclusterlist[i])
    return result


# 话题预测追踪
def topicPredict(new_text_data: list, existing_topics: pd.DataFrame):
    '''
    From existing topics to predict topics of new text data with SVM.
    
    input:
        new_text_data: List of new documents.
        existing_topics: DataFrame contains topic and documents belongs to the topic, in plain text.
    
    output:
        topics: topics and probabilities of new documents.
    '''
    # 已有的热点话题数据
    
    # 标签，即已有的话题
    print(f"已有的热点话题为{list(set(existing_topics['topic'].tolist()))}")
    
    # 标签对应的类别
    existing_labels = existing_topics['num_']
    print(f"总共有类别数：{len(list(set(existing_labels.tolist())))}")

    # 特征提取（TF-IDF）
    # vectorizer = TfidfVectorizer()
    # X_existing = vectorizer.fit_transform(existing_topics)

    # 将文本数据转换为词袋模型特征向量
    vectorizer = CountVectorizer()
    X_existing = vectorizer.fit_transform(existing_topics['doc'].tolist())
    X_new = vectorizer.transform(new_text_data)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_existing, existing_labels, test_size=0.2, random_state=42)

    # 训练SVM模型
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 对新的话题数据进行趋势预测
    new_topic_predictions = svm_model.predict(X_new)

    # 输出预测结果
    topics = []
    for i in range(len(new_topic_predictions)):
        topics.append((new_topic_predictions[i], svm_model.predict_proba(X_new[i])[0][new_topic_predictions[i]]))
        
    return topics