from Preprocess import *
from modelProcess import *
import pandas as pd

from KmeansTopic import *
from SVMPredict import topicPredict

import matplotlib
import matplotlib.pyplot as plt
from pylab import xticks, yticks, np
from gensim import models
import gensim.corpora as corpora

import pyLDAvis.gensim

# Procedure 1 - Split with time
def splitWithTime(df: pd.DataFrame, time_interval: int, lang: str = "zh-cn"):
    '''
    Format the time column and split to time intervals.
    
    input:
        df: Raw data need to process.
        time_interval: Time interval for splition.
        lang: Language used in the procession.
        
    output:
        list_df: List of dataframe after splition.
    '''
    # time processing.
    if lang == "zh-cn":
        df, time_slices = processTime_cn(df, time_interval)
    if lang == "en-us":
        df, time_slices = processTime_en(df, time_interval)
    
    list_df = []
    
    length = len(time_slices)
    # time splition.
    for i in range(length):
        # judge if until now
        if i == length - 1:
            df_component = df[df['time'] >= time_slices[i]]
        else:
            df_component = df[(df['time'] >= time_slices[i]) & (df['time'] <= time_slices[i + 1])]
        list_df.append({
            "df": df_component,
            "time": time_slices[i]
        })
        
    return list_df
        
    

# Procedure 2 - Preprocess
def preprocess(df: pd.DataFrame, stopwords: list = None, lang: str = "zh-cn"):
    '''
    Do the preprocess procedure.
    
    input:
        df: Raw data needs preprocession. 
        stopwords: Stopwords used in preprocession, only for en-us stopwords could be None.
        lang: Language used in the procession.
    
    output:
        df: Data after preprocession.
    '''
    df = dataProcess(df, stopwords, lang)
    
    return df

# Procedure 3 - Topic extraction with BoW + TF-IDF + LDA
def TopicExtraction(textList: list, num_topics: int):
    '''
    Do the topic extraction process.
    
    input:
        textList: List of texts need to extract the topic.
        num_topics: Number of topics to extract.
        
    output:
        model_list: List of lda models, for manual selection.
        dic: Dictionary for this textlist.
        corpus_tfidf: Corpus after tfidf.
    '''
    # BoW
    dic, corpus = BoW(textList)
    
    # tf-idf
    corpus_tfidf = TFIDF(dic=dic, corpus=corpus)
    
    # LDA
    model_list, x, perplexity, coherence = LDA(strList=textList,
                                               corpus_tfidf=corpus_tfidf,
                                               dic=dic,
                                               num_topics=num_topics)
    
    
    # draw figure
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False 

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity, marker="o")
    plt.title("Perplexity")
    plt.xlabel('num_topics')
    plt.ylabel('Perplexity')
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True)) # 保证x轴刻度为1

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence, marker="o")
    plt.title("Coherence")
    plt.xlabel("num_topics")
    plt.ylabel("Coherence")
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))
    
    plt.show()
    
    return model_list, dic, corpus_tfidf
    
# Procedure 4 - hotSpot analysis using KMeans and topics
def hotSpotAnalysis(lda_model: models.ldamodel.LdaModel, df: pd.DataFrame, dic: corpora.Dictionary,topic_nums: int, num_words: int, emotional_dict_path: str):
    '''
    Analyse the hotspot topics and extract hotspot topics.
    
    input:
        lda_model: Model with the parameters from manual selection.
        df: Data after preprocession, for doc2bow.
        dic: Dictionary for this textlist.
        topic_nums: Topic counts with manual selection.
        num_words: Manual selected topic size to output.
        
    output:
        all_top_k: Top-k hot words in each topic clustered by KMeans.
        doc_indexs: Document indexs belongs to that cluster of hot words.
    '''
    # Get topics.
    topics = lda_model.print_topics(num_words=num_words)
    
    # Get topic distribution of documents.
    zeros = np.zeros(shape=[df.shape[0],topic_nums])
    doc_dist = pd.DataFrame(zeros, columns=[str(i) for i in range(0, topic_nums)], index=range(df.shape[0]))
    i = 0
    for each in df['contents']:
        # 获取每篇文档的主题分布，并添加入特征矩阵
        doc_bow = dic.doc2bow(each) # doc2bow
        doc_lda = lda_model[doc_bow]
        for each in doc_lda[0]:
            doc_dist[str(each[0])][i] = each[1]
        i += 1
    
    # KMeans clustering.
    clusters, n_clusters = KMeansGrouping(doc_dist)
    hot_words = hotSpot(cluster=clusters, topic=topics, dist=doc_dist, n=n_clusters)
    
    # Sentiment analysis.
    all_top_k, doc_indexs = topic_analyze(hot_words, emotional_dict_path)
    
    return all_top_k, doc_indexs

# Procedure 5 - Visualization
def visualize(lda_model: models.ldamodel.LdaModel, corpus_tfidf: list, dic: corpora.Dictionary, pyLDAvis_save_path: str, all_top_k: list, wordCloud_pic_path: str, wordCloud_font_path: str, wordCloud_save_path: str):
    '''
    Visualize with pyLDAvis and wordCloud.
    
    input:
        lda_model: Model with the parameters from manual selection.
        corpus_tfidf: Corpus after tfidf.
        dic: Dictionary for this textlist.
        pyLDAvis_save_path: Path for html created by pyLDAvis.
        all_top_k: Top-k hot words in each topic clustered by KMeans.
        wordCloud_pic_path: Picture path for wordCloud.
        wordCloud_font_path: Font path for wordCloud.
        wordCloud_save_path: Path for pictures created by wordCloud.
        
    output:
        None
    '''
    # pyLDAvis可视化
    data = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus_tfidf, dictionary=dic)
    pyLDAvis.save_html(data, pyLDAvis_save_path)
    
    # 词云
    wordCloud(all_top_k=all_top_k, pic_path=wordCloud_pic_path, font_path=wordCloud_font_path, save_path=wordCloud_save_path)
    
    return

# Procedure 6 - Prediction
def merge(all_top_k: list, doc_indexs: list):
    '''
    Merge similar topics in all_top_k.
    
    input:
        all_top_k: Top-k hot words in each topic clustered by KMeans.
        doc_indexs: Document indexs belongs to that cluster of hot words.
    
    output:
        topic_doc: key-value pairs of topic and docs belongs to the topic.
    '''
    assert(len(all_top_k) == len(doc_indexs))
    topic_doc = dict()
    for i in range(len(all_top_k)):
        if all_top_k[i] not in topic_doc:
            topic_doc[all_top_k[i]] = doc_indexs[i]
        else:
            topic_doc[all_top_k[i]] += doc_indexs[i]
    return topic_doc

def SVM_Predict(new_documents: list, topic_doc: dict, doc_list: list):
    '''
    Generate predictions for newly add documents.
    
    input:
        new_documents: List of texts after preprocession.
        topic_doc: key-value pairs of topic and docs belongs to the topic.
        doc_list: Document list for training.
    
    output:
        topic: List of which topic each document belongs to and probabilities for the prediction.
    '''
    
    # generate training data.
    # generate the topic-index pattern.
    train_ = []
    num_ = 0
    for key, value in topic_doc.items():
        temp = dict()
        temp['topic'] = key
        temp['doc'] = value
        temp['num_'] = num_
        num_ += 1
        train_.append(temp)
    
    train = pd.DataFrame(train_, columns=['topic', 'doc', 'num_'])
    train = train.explode(column='doc').reset_index().drop(columns='index')
    
    # transform index into document.
    doc_new = []
    for each in train['doc']:
        doc_new.append(" ".join(doc_list[each]))
    train['doc'] = doc_new
    
    topic = topicPredict(new_documents, train)
    
    return topic

# Procedure 7 - Judge predictions
def Judge(topic: list, threshold: int, trust: float):
    '''
    Judge if there appears new topic and need to rerun LDA.
    
    input:
        topic: topic and probabilities generated from SVM.
        threshold: threshold for judging new topics.
        trust: if document probability less than trust then it may be new topic.
        
    output:
        None.
    '''
    num_ = 0
    for each in topic:
        if each[1] < trust:
            num_ += 1
    if num_ > threshold:
        print("Maybe new topic merges, need to rerun LDA.")
    else:
        print("Prediction done, no new topics merged.")