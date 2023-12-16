import os
from os import path

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
# from exam import lda
from emotionAnalysis import getScore

from gensim import models

import re

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

# KMeans聚类，并输出以原始文本表示的聚类结果
def KMeansGrouping(dist: pd.DataFrame) -> (list, int):
    # # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的 词频
    # vectorizer = CountVectorizer(max_features=20000)
    # # 该类会统计每个词语的tf-idf权值
    # tf_idf_transformer = TfidfTransformer()
    # # 将文本转为词频矩阵并计算tf-idf
    # tfidf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(corpus))
    # # 获取词袋模型中的所有词语
    # tfidf_matrix = tfidf.toarray()
    # print(f"tfidf:{tfidf_matrix}")

    # 假设数据为 X
    # X = ...
    # K-means聚类方法开始
    # 存储每一轮的best_k
    best_k_num = []
    # 尝试不同的 K 值
    for d in range(0, 4):
        # 存储每个 K 对应的轮廓系数
        silhouette_scores = []
        for k in range(2, 9):  # 选择一个合适的 K 范围
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(dist)
            silhouette_avg = silhouette_score(dist, labels)
            # 将每一轮的轮廓系数保存
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

        # 根据轮廓系数选择最优的 K
        best_k = np.argmax(silhouette_scores) + 2  # 加2是因为从K=2开始
        print(f"Best K value based on silhouette score: {best_k}")
        best_k_num.append(best_k)

    best_k_mean_value = np.mean(best_k_num)
    print(f"bestk mean value = {math.floor(best_k_mean_value)}")

    # 找到最优的K值后，对数据文本进行聚类
    clf = KMeans(n_clusters=math.floor(best_k_mean_value))
    # 存储 KMeans 模型的拟合结果
    s = clf.fit(dist)
    # # 每个样本所属的簇
    # label = []
    # i = 1
    # while i <= len(clf.labels_):
    #     label.append(clf.labels_[i - 1])
    #     i = i + 1
    # 获取标签聚类
    y_pred = clf.labels_
    # print(y_pred)

    # pca降维，将数据转换成二维
    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(dist)  # 载入N维

    xs, ys = newData[:, 0], newData[:, 1]
    # 设置颜色
    cluster_colors = {0: 'r', 1: 'yellow', 2: 'b', 3: 'chartreuse', 4: 'purple', 5: '#FFC0CB', 6: '#6A5ACD',
                      7: '#98FB98', 8: 'brown', 9: 'pink', 10: 'blue'}

    # 设置类名
    cluster_names = {0: u'类0', 1: u'类1', 2: u'类2', 3: u'类3', 4: u'类4', 5: u'类5', 6: u'类6', 7: u'类7', 8: u'类8', 9: u'类9', 10: u'类10',   }
    # 绘制散点图
    df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred))
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(8, 5))  # set size
    ax.margins(0.02)
    for name, group in groups:
        ax.plot(np.array(group.x), np.array(group.y), marker='o', linestyle='', ms=10, label=cluster_names[name],
                color=cluster_colors[name], mec='none')
    plt.show()
    # 将聚类结果转换为原始文本
    res = y_pred

    return res, math.floor(best_k_mean_value)

# 计算簇内词分布并提取热点话题
def hotSpot(cluster: np.ndarray, topic: list, dist: pd.DataFrame, n: int):
    # 计算簇内文档数，并记录簇内文档的索引
    docs = []
    doc_index = []
    for i in range(n):
        docs.append(cluster.tolist().count(i))
        doc_index.append(np.where(cluster==i)[0].tolist())
    
    # 处理topic
    topic_list = dict()
    for each in topic:
        temp_1 = re.sub(r'[ "]+',"", each[1])
        temp_2 = re.split(r'[*+]+', temp_1)
        kv = dict()
        i = 0
        while (i < len(temp_2)):
            kv[temp_2[i + 1]] = float(temp_2[i])
            i += 2
        topic_list[each[0]] = kv
    
    # 计算词汇似然概率
    cluster_words = dict()
    for index, row in dist.iterrows():
        if cluster[index] not in cluster_words:
            cluster_words[cluster[index]] = dict()
        for topic_num in range(0, row.size):
            if row[topic_num] <= 0.0:
                continue
            else:
                current_topic = topic_list[topic_num]
                for key, value in current_topic.items():
                    if key not in cluster_words[cluster[index]]:
                        cluster_words[cluster[index]][key] = value * row[topic_num]
                    else:
                        cluster_words[cluster[index]][key] += value * row[topic_num]
                        
    # 排序并输出
    hot_spots = []
    for clusts, hotwords in cluster_words.items():
        sort_result = sorted(hotwords.items(), key=lambda x: -x[1])
        hot_spots.append((docs[clusts], [i[0] for i in sort_result[:10]], doc_index[clusts]))
    
    return hot_spots
    
# 热点话题分析
def topic_analyze(hot_spots: list, emotional_dict_path: str):
    hot_spots.sort(key=lambda x: -x[0])
    all_top_k = []
    doc_indexs = []
    for each in hot_spots:
        top_k_features = each[1]
        doc_num = each[0]
        print("热点话题: {}, 属于此话题的文档数: {}".format(top_k_features, doc_num))
        flattened_string = ' '.join(top_k_features)
        print("字符串热点话题:", flattened_string)
        # 情感分析
        score = getScore(flattened_string, emotional_dict_path)
        print(f"情感分析得分：{score}")
        all_top_k.append(flattened_string)
        doc_indexs.append(each[2])
    return all_top_k, doc_indexs
        

# 对聚类结果进行分析，返回所有的热点话题
def analyse(res: list, emotional_dict_path: str):
    # 获取每个簇的文本内容
    mum = 0
    vectorizer = CountVectorizer(max_features=1000, stop_words=None)
    all_top_k = []
    all_emotionScore = []
    for cluster_label in range(len(res)):  # num_clusters 为聚类的数量
        cluster_texts = []
        print(f"mun=  {mum}")
        for cluster_text in res[cluster_label]:
            cluster_texts.append(cluster_text)
        # 使用 CountVectorizer 进行词频统计
        print(f"cluster text: {cluster_texts}")
        if len(cluster_texts) > 0:
            tf_matrix = vectorizer.fit_transform(cluster_texts)
            # 获取词汇列表
            feature_names = vectorizer.get_feature_names_out()
            # 统计词频
            word_freq = tf_matrix.sum(axis=0)
            # 找到词频最高的词语作为热点话题
            top_k = 10  # 假设选择前 10 个热点话题
            top_k_features = [feature_names[i] for i in word_freq.argsort()[0, -top_k:]]
            print("热点话题:", top_k_features)
            # flattened_string = ' '.join(top_k_features)
            top_k_features = np.array(top_k_features)
            # flattened_string = ' '.join([' '.join(row) for row in np.array(top_k_features)])
            print("字符串热点话题:", top_k_features)
            flattened_string = ' '.join([' '.join([' '.join(item) for item in row]) for row in top_k_features])
            print("字符串热点话题:", flattened_string)
            # 情感分析
            score = getScore(flattened_string, emotional_dict_path)
            print(f"情感分析得分：{score}")
            all_emotionScore.append(score)
            all_top_k.append(flattened_string)
            mum = mum + 1
        else:
            break
    return all_top_k

# 生成词云
def wordCloud(all_top_k: list, pic_path: str, font_path: str, save_path: str):
    # 生成词云
    final_all_top_k = ' '.join(np.array(all_top_k))
    print("final字符串热点话题:", final_all_top_k)
    pic_mask = np.array(Image.open(pic_path))
    # alice_mask = np.array(Image.open(os.path.abspath(os.path.join(d, "alice_mask.png"))))
    stopwords = set(STOPWORDS)
    stopwords.add("said")
    # 设置中文字体
    # font_path = r"data/SourceHanSerif-VF.otf.ttc"  # 请根据系统实际路径修改
    wc = WordCloud(background_color="white", max_words=2000, mask=pic_mask, font_path=font_path,
                   stopwords=stopwords, contour_width=3, contour_color='steelblue')
    # generate word cloud
    wc.generate(final_all_top_k)
    # store to file
    wc.to_file(save_path)
    # show
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.imshow(wc, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.show()