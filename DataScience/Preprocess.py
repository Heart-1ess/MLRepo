import jieba
import jieba.analyse
import jieba.posseg

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from dateutil.parser import parse

import re
import pandas as pd
import numpy as np

import string

# ----------------------------- zh-cn ------------------------------------

# 文本清洗
def clearText_cn(line: str) -> str:
    '''
    Clean the data, delete numbers and letters.
    input: 
        line: data needs preprocess.
    
    output:
        line: data after preprocess, chinese plain text only.
    
    '''
    if (line != ''):
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）‘’]+", "", line)
        return line
    return None

# 文本切割并仅保留名词、去停用词
def sent2word_cn(line: str, stopwords: list) -> list:
    '''
    Use Jieba to split the words.
    input:
        line: data after preprocess but needs splition.
    
    output:
        segSentence: data after splition, as list format.
    '''
    jieba.suggest_freq(('人权', '人道', '人民', '人口'), tune=True)
    segList = jieba.posseg.cut(line)
    segSentence = []
    word_list = ['ns', 'n', 'nz', 'vi', 'v', 'nf', 'nr', 'nt', 'nl', 'ng']
    for x in segList:
        if x.flag in word_list and x.word != '\t' and not ifstopWords(x.word, stopwords):
            segSentence.append(x.word)
    return segSentence

# 判断停用词
def ifstopWords(word: str, stopwords: list) -> bool:
    '''
    Delete the stopwords for better performance.
    input:
        word: word for judgement.
    
    output:
        res: if the word is stopword.
    '''
    if word in stopwords:
        return True
    return False

# 读入停用词
def readStopWords(filepath: str) -> list:
    '''
    Read in the stopwords from stopwords list.
    input: 
        filepath: filepath for the stopwords list.
    
    output:
        stopwords: list of stopwords extract from the stopwords list.
    '''
    stopwords = []
    # Read in txt.
    with open(filepath, "r", encoding="GBK") as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords

# 处理时间信息，并按时间片划分
def processTime_cn(df: pd.DataFrame, time_interval: int) -> (pd.DataFrame, list):
    '''
    Process the time information (from CN to yyyy-mm-dd method and split to time_slices with time_interval)
    
    input:
        df: DataFrame to process, contains a column named time.
        time_interval: time interval to slice rows, default as days.
    
    output:
        df: DataFrame after process.
        time_slices: time slices after splition. Use for ldaseqModel.
    '''
    # Turn to datetime from string pattern.
    time = df['time']
    ymd = ['{}-{}-{}'.format(re.split('年|月|日', i)[0], re.split('年|月|日', i)[1], re.split('年|月|日', i)[2]) for i in time]
    ymd = pd.to_datetime(ymd)
    df['time'] = ymd
    
    # sort with time ascending and slice with intervals.
    df = df.sort_values('time', ascending=True)
    df = df.reset_index()
    time_slices = []
    time_now = df['time'][0]
    while (df['time'].tolist()[-1] > time_now):
        time_slices.append(time_now)
        time_now += pd.DateOffset(days=time_interval)
    return df, time_slices

# ---------------------------------- en-us ----------------------------------

# 文本清洗
def clearText_en(text: str) -> str:
    '''
    Clean the data, delete numbers and letters.
    input: 
        text: data needs preprocess.
    
    output:
        text: data after preprocess, english plain text only.
    
    '''
    # 缩写清洗
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"I'm","I am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"He's","He is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"SHe's","She is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"this's","this is",text)
    text=re.sub(r"it's","it is",text)
    text=re.sub(r"here's","here is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"who's","who is",text)
    text=re.sub(r"how's","how is",text)
    text=re.sub(r"let's","let us",text)
    text=re.sub(r"That's","That is",text)
    text=re.sub(r"This's","This is",text)
    text=re.sub(r"It's","It is",text)
    text=re.sub(r"Here's","Here is",text)
    text=re.sub(r"What's","What is",text)
    text=re.sub(r"Where's","Where is",text)
    text=re.sub(r"Who's","Who is",text)
    text=re.sub(r"How's","How is",text)
    text=re.sub(r"Let's","Let us",text)
    text=re.sub(r"you're","you are",text)
    text=re.sub(r"You're","You are",text)
    text=re.sub(r"we're","we are",text)
    text=re.sub(r"We're","We are",text)
    text=re.sub(r"they're","they are",text)
    text=re.sub(r"They're","They are",text)
    text=re.sub(r"there's","there is",text)
    text=re.sub(r"there're","there are",text)
    text=re.sub(r"There's","There is",text)
    text=re.sub(r"There're","There are",text)
    text=re.sub(r"don't","do not",text)
    text=re.sub(r"doesn't","does not",text)
    text=re.sub(r"can't","can not",text)
    text=re.sub(r"couldn't","could not",text)
    text=re.sub(r"Don't","Do not",text)
    text=re.sub(r"Doesn't","Does not",text)
    text=re.sub(r"Can't","Can not",text)
    text=re.sub(r"Couldn't","Could not",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"wouldn't","would not",text)
    text=re.sub(r"shouldn't","should not",text)
    text=re.sub(r"Won't","Will not",text)
    text=re.sub(r"Wouldn't","Would not",text)
    text=re.sub(r"Shouldn't","Should not",text)
    text=re.sub(r"hasn't","has not",text)
    text=re.sub(r"haven't","have not",text)
    text=re.sub(r"isn't","is not",text)
    text=re.sub(r"wasn't","was not",text)
    text=re.sub(r"aren't","are not",text)
    text=re.sub(r"weren't","were not",text)
    text=re.sub(r"didn't","did not",text)
    text=re.sub(r"ain't","am not",text)
    text=re.sub(r"Hasn't","Has not",text)
    text=re.sub(r"Haven't","Have not",text)
    text=re.sub(r"Isn't","Is not",text)
    text=re.sub(r"Wasn't","Was not",text)
    text=re.sub(r"Aren't","Are not",text)
    text=re.sub(r"Weren't","Were not",text)
    text=re.sub(r"Didn't","Did not",text)
    text=re.sub(r"Ain't","Am not",text)
    text=re.sub(r"i've",'i have',text)
    text=re.sub(r"you've",'you have',text)
    text=re.sub(r"he've",'he have',text)
    text=re.sub(r"she've",'she have',text)
    text=re.sub(r"we've",'we have',text)
    text=re.sub(r"they've",'they have',text)
    text=re.sub(r"would've",'would have',text)
    text=re.sub(r"could've",'could have',text)
    text=re.sub(r"should've",'should have',text)
    text=re.sub(r"I've",'I have',text)
    text=re.sub(r"You've",'You have',text)
    text=re.sub(r"He've",'He have',text)
    text=re.sub(r"She've",'She have',text)
    text=re.sub(r"We've",'We have',text)
    text=re.sub(r"They've",'They have',text)
    text=re.sub(r"Would've",'Would have',text)
    text=re.sub(r"Could've",'Could have',text)
    text=re.sub(r"Should've",'Should have',text)
    text=re.sub(r"i'll",'i will',text)
    text=re.sub(r"you'll",'you will',text)
    text=re.sub(r"he'll",'he will',text)
    text=re.sub(r"she'll",'she will',text)
    text=re.sub(r"it'll",'it will',text)
    text=re.sub(r"we'll",'we will',text)
    text=re.sub(r"they'll",'they will',text)
    text=re.sub(r"I'll",'I will',text)
    text=re.sub(r"You'll",'You will',text)
    text=re.sub(r"He'll",'He will',text)
    text=re.sub(r"She'll",'She will',text)
    text=re.sub(r"It'll",'It will',text)
    text=re.sub(r"We'll",'We will',text)
    text=re.sub(r"They'll",'They will',text)
    text=re.sub(r"i'd","i would",text)
    text=re.sub(r"you'd","you would",text)
    text=re.sub(r"he'd","he would",text)
    text=re.sub(r"she'd","she would",text)
    text=re.sub(r"it'd","it would",text)
    text=re.sub(r"we'd","we would",text)
    text=re.sub(r"they'd","they would",text)
    text=re.sub(r"I'd","I would",text)
    text=re.sub(r"You'd","You would",text)
    text=re.sub(r"He'd","He would",text)
    text=re.sub(r"She'd","She would",text)
    text=re.sub(r"It'd","It would",text)
    text=re.sub(r"We'd","We would",text)
    text=re.sub(r"They'd","They would",text)
    #print('replace_contractions done!')
    
    # 代词清洗
    text=re.sub(f'<.*>',' ',text)
    punc=string.punctuation
    p=re.compile(f'[{punc}]+')
    text=re.sub(p,' ',text)
    #print('remove_pron done!')
   
    # 数字和url清洗
    text=re.sub(r'https?://\S+',' ',text)
    text=re.sub('[0-9]+[\.]?[0-9]+',' ',text)
    text=re.sub('@[\S]+',' ',text)
    #print('remove_number_user_url done!')

    # 多余空格清洗
    text=re.sub('[^A-Za-z]+',' ',text)
    text=re.sub('[\s]+',' ',text)
    #print('remove_extra_blank done!')
    
    return text

def stopwordslist(lemmatized_words: list, stop_words: set) -> list:
    '''
    Delete the stopwords for better performance.
    
    input:
        lemmatized_words: Words list after tokenization.
        stop_words: Set of stopwords.
    
    output:
        filtered_list: Words list after deletion of stopwords.
    '''
    # 创建一个空列表
    filtered_list = []
    for word in lemmatized_words:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    return filtered_list

def sent2word_en(corpus: str, stop_words: set) -> list:
    '''
    Use NLTK to split the words.
    input:
        corpus: data after preprocess but needs splition.
        stop_words: stopwords to delete.
    
    output:
        nn_nns_words: data after splition, as list format.
    '''
    # 分词
    words = word_tokenize(corpus)
    
    # 去停用词
    filtered_list = stopwordslist(words, stop_words)
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_list]
    
    # 词性标记
    new_leg = nltk.pos_tag(filtered_list)
    
    # 取一般名词
    nn_nns_words = [word for word, tag in new_leg if tag in ['NNS', 'NN']]
    
    return nn_nns_words

def processTime_en(df: pd.DataFrame, time_interval: int) -> (pd.DataFrame, list):
    '''
    Process the time information (from CN to yyyy-mm-dd method and split to time_slices with time_interval)
    
    input:
        df: DataFrame to process, contains a column named time.
        time_interval: time interval to slice rows, default as days.
    
    output:
        df: DataFrame after process.
        time_slices: time slices after splition. Use for ldaseqModel.
    '''
    # for each element apply parsing method.
    df['time'].apply(parse)
    
    # sort with time ascending and slice with intervals.
    df = df.sort_values('time', ascending=True)
    df = df.reset_index()
    time_slices = []
    time_now = df['time'][0]
    while (df['time'][-1] > time_now):
        time_slices.append(time_now)
        time_now += pd.DateOffset(days=time_interval)
    return df, time_slices
    

# ---------------------------------- process ---------------------------------

# 处理DataFrame
def dataProcess(df: pd.DataFrame, stopwords: list = None, lang: str = "zh-cn") -> pd.DataFrame:
    '''
    Process the structured data from pd.DataFrame.
    input: 
        df: DataFrame to process, contains a column named contents.
        stopwords: Path of stopwords if need, default in None.
        time_interval: Time interval for slicing.
        lang: language of documents, default in zh-cn.
        
    output:
        df: DataFrame after procession.
        time_slices: 
    '''
    contents = df['contents']
    contents_new = []
    
    # 中文处理
    if (lang == "zh-cn"):
        for each in contents:
            # clear the text.
            temp_1 = clearText_cn(each)
            # split the words and preserve nouns, also delete stopwords.
            temp_2 = sent2word_cn(temp_1, stopwords)
            contents_new.append(temp_2)
        df['contents'] = contents_new
        
        return df
    
    # 英文处理
    elif (lang == "en-us"):
        # read in English stopwords.
        stop_words = set(stopwords.words("english"))
        
        # start processing.
        for each in contents:
            # clear the text.
            temp_1 = clearText_en(each)
            
            # split the words and preserve nouns, also delete stopwords.
            temp_2 = sent2word_en(temp_1, stop_words)
            contents_new.append(temp_2)
        df['contents'] = contents_new
        
    return df

def processTxt(data: str, lang: str = "zh-cn", stopwords: str = None) -> str:
    '''
    Process newly added text and do the splition.
    
    input:
        data: newly added text.
        lang: language the text used.
    
    output:
        temp_2: data after procession.
    '''
    
    if (lang == "zh-cn"):
        temp_1 = clearText_cn(data)
        temp_2 = sent2word_cn(temp_1, stopwords)
        
    elif (lang == "en-us"):
        stop_words = set(stopwords.words("english"))
        temp_1 = clearText_en(data)
        temp_2 = sent2word_en(temp_1, stop_words)
        
    return temp_2
    
    
    
if __name__ == "__main__":
    # 读取csv
    df = pd.read_csv('/home/chengyuli/yanshan/data/DataScience/UN_raw.csv')
    # 读取停用词表
    stopwordsPath = "/home/chengyuli/yanshan/data/DataScience/stop_words_ch.txt"
    stopwords = readStopWords(stopwordsPath)
    # 进行数据处理
    # print(dataProcess(df, stopwords=stopwords).head())
    # print(processTime(df, 7)[1])