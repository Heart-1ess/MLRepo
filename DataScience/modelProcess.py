import gensim.corpora as corpora
from gensim import models
import os
import math

# 词典与词袋语料库创建
def BoW(strList: list, dicPath: str = None) -> (corpora.Dictionary, list):
    '''
    Use BoW transform words to dictionaries.
    input:
        strList: list of words after preprocession.
        
    output:
        dic: dictionary build with strList.
        corpus: list of bag vectors of each article.
    '''
    dic = corpora.Dictionary(strList)
    # 出现次数至少为3，至多在一半文档中出现，词典大小最多3000个词
    dic.filter_extremes(no_below=1, no_above=0.8, keep_n=3000)
    
    # 保存词典
    if dicPath:
        dic.save_as_text(dicPath)
    
    # 创建语料库
    corpus = [dic.doc2bow(text) for text in strList]
    
    return dic, corpus

# 词袋用TF-IDF格式化为权重
def TFIDF(dic: corpora.Dictionary, corpus: list, model_path: str = None) -> list:
    '''
    Use TF-IDF model to formatize bag vectors to weights.
    input:
        dic: dictionary built before.
        corpus: list of bag vectors of each article.
        model_path: model path to read or save model.
    
    output:
        corpus_tfidf: list of bag vectors after formatizing.
    '''
    # 判断是否存在，如果模型存在就读入，否则保存
    if model_path and os.path.isfile(model_path):
        tfidf_model = models.TfidfModel.load(model_path)
    elif model_path:
        tfidf_model = models.TfidfModel(corpus=corpus, dictionary=dic)    
        tfidf_model.save(model_path)
    else:
        tfidf_model = models.TfidfModel(corpus=corpus, dictionary=dic)
    
    # TF-IDF格式化为权重
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    
    return corpus_tfidf

# 适用LDA主题模型进行主题提取
def LDA(strList: list, corpus_tfidf: list, dic: corpora.Dictionary, num_topics: int) -> list:
    '''
    Use LDA model to extract the topics and display.
    input:
        corpus_tfidf: list of bag vectors after tf-idf formatizing.
        dic: dictionary built before.
        num_words: number of words to extract.
        num_topics: number of topics to train.
        
    output:
        result: topK topics written with param num_words.
    '''
    # 创建模型
    model_list = []
    perplexity = []
    coherence = []
    x = [] # x轴
    
    # 调参
    for topic in range(num_topics):
        print('num_topics = {}'.format(topic + 1))
        lda_model = models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                             id2word=dic,
                                             num_topics=topic+1,
                                             random_state=10,
                                             update_every=1,
                                             chunksize=100,
                                             passes=10,
                                             alpha='auto',
                                             per_word_topics=True)
        model_list.append(lda_model)
        x.append(topic + 1)
        
        # bound, 困惑度为2^(-log_perplexity)
        p = lda_model.log_perplexity(corpus_tfidf)
        perp = math.pow(2, -p)
        perplexity.append(perp)
        print("Perplexity = {}".format(perp))
        
        coherenceModel = models.CoherenceModel(model=lda_model, texts=strList, dictionary=dic, coherence='c_v')
        c = coherenceModel.get_coherence()
        coherence.append(c)
        print('Coherence = {}'.format(c))
    
    return model_list, x, perplexity, coherence