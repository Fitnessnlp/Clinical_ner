#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/23 15:09
# @Author: chenwei
# @File  : main_model.py

import numpy as np
import pandas as pd
import kashgari
import re
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from kashgari.callbacks import EvalCallBack
from Bare_BiLSTM_CRF import processing_data
import codecs
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

def read_text(path):
    text = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        t = f.readline()
        text.append(t)
    return text

def preprocess_text(text):
    #text = text.lower()                                     # 转为小写
    text = re.sub('[^a-z^0-9^\u4e00-\u9fa5]', '', text)     # 去除标点，这里只训练中英文字符
    text = re.sub("",'',text) # 去掉不可识别的东西
    #text = re.sub('[0-9]', '0', text)                       # 将所有数字都转为0
    return text

def w2v_data(data_file, corpus_path):

    with open(data_file, 'r', encoding='utf-8') as f:
        total_sent = []
        for line in f.readlines():
            # print(line)
            line = preprocess_text(line)
            # print(line)
            sent = ' '.join(line)
            total_sent.append(sent)
    with codecs.open(corpus_path, 'w', encoding='utf-8') as f1:
        f1.write('\n'.join(total_sent))

    with codecs.open(corpus_path, 'r', encoding='utf-8') as f2:
        sentences = [line.split(' ') for line in f2.read().split('\n')]

    return sentences

def text_split(text):
    # 连续多个空格，合并为一个
    text = re.sub(r'\s+',' ', text)
    # 去除空格
    text = re.sub(' ', '', text)
    #text = self.text_filter(text)
    text1 = re.split(r'。', text)

    return text1

def pos2ners(s, pos):
    def pos2ner_item(Type):
        result = []
        if 'B-' + Type in pos:
            postions = [i for i, v in enumerate(pos) if v == 'B-' + Type]
            for p in postions:
                index_start = p
                index_end = index_start
                for i in range(index_start + 1, len(pos)):
                    if pos[i] != 'I-' + Type:
                        break
                    index_end = i
                result.append(s[index_start:index_end + 1])
                pass
            pass
        return result
        pass

    result = {}
    # 原发部位
    Ys = pos2ner_item('ORI')
    # 病灶大小
    Ss = pos2ner_item('SIZ')
    # 转移部位
    Zs = pos2ner_item('TRA')
    result['ORI'] = Ys
    result['SIZ'] = Ss
    result['TRA'] = Zs
    return result

def list_to_str(L):
    ss = ""
    for i in range(len(L)):
        ss += L[i]
        if (i != len(L) - 1):
            ss += ','
    return ss

def predi_output(data, typ, loaded_model):

    # Word2Vec
    if typ == 'W2V':
        texts = text_split(data)
        sent = texts
    else: # Bert embedding
        texts = text_split(data)
        #texts = list(texts)
        sent = []
        for t in texts:
            sent.append(list(t))

    result = loaded_model.predict(sent)
    Y = list()
    S = list()
    Z = list()
    for i in range(len(texts)):
        sentence = texts[i]
        # if len(sentence) != len(texts[i]):
        # print(sentence, texts[i])
        # print(len(sentence), len(texts[i]))
        pos = pos2ners(sentence, result[i])
        if len(pos['ORI']) != 0:
            for j in pos['ORI']:
                Y.append(j)
        if len(pos['SIZ']) != 0:
            for j in pos['SIZ']:
                S.append(j)
        if len(pos['TRA']) != 0:
            for j in pos['TRA']:
                Z.append(j)
        Y = list(set(Y))
        S = list(set(S))
        Z = list(set(Z))
    Y_str = list_to_str(Y)
    S_str = list_to_str(S)
    Z_str = list_to_str(Z)
    return Y_str, S_str, Z_str


if __name__ == '__main__':
    '''1000条非标注数据训练领域的词向量'''
    data_file = './data/unlabeled.txt'
    corpus_path = './data/corpus.txt'
    sentences = w2v_data(data_file, corpus_path)
    path = get_tmpfile("word2vec.model")
    # 用word2Vec做的词嵌入，语料是未标注的1000条各个场景数据
    model = Word2Vec(sentences, size=100, window=15, min_count=1, workers=4)
    model.wv.save_word2vec_format("word2vec.model", binary=True)

    '''NER训练部分'''
    datatwo = pd.read_excel('./data/BLSTMCRF_DATA1.xlsx')
    test_data = pd.read_excel('./data/test.xlsx')
    #data1 = pd.read_excel('C:/Users/Caesar/Desktop/NER_Chinese/data/training_part1.xlsx')
    #data2 = pd.read_excel('C:/Users/Caesar/Desktop/NER_Chinese/data/training_part2.xlsx')
    #data = pd.concat((data1, data2), axis=0, ignore_index=True)
    #result = processing_data.get_bio_pos(datatwo)
    result = processing_data.get_bio_pos(datatwo)
    result = np.array(result)
    result_test = processing_data.get_bio_pos(test_data)
    result_test = np.array(result_test)
    # train_x, test_x, train_y, test_y = train_test_split(result[:,0],  result[:,1], test_size=0.3, random_state=0)
    train_x = result[:, 0]
    train_y = result[:, 1]
    test_x = result[:, 0]
    test_y = result[:, 1]
    train_x = list(train_x)
    train_y = list(train_y)
    test_x = list(test_x)
    test_y = list(test_y)

    ''' BERT Embedding '''
    #embedding = BERTEmbedding('./chinese_L-12_H-768_A-12',
    #                             task = kashgari.LABELING,
    #                             sequence_length = 150)
    ''' Word2Vec Embeddings '''
    word2vec_embedding = kashgari.embeddings.WordEmbedding(w2v_path="word2vec.model",
                                                           task=kashgari.LABELING,
                                                           w2v_kwargs={'binary': True, 'unicode_errors': 'ignore'},
                                                           sequence_length='auto')
    model = BiLSTM_CRF_Model(word2vec_embedding)
    #model = BiLSTM_CRF_Model(embedding)
    tf_board_callback = keras.callbacks_v1.TensorBoard(log_dir='.\\logs', update_freq=1000)
    eval_callback = EvalCallBack(kash_model=model,
                                 valid_x=test_x,
                                 valid_y=test_y,
                                 step=4)

    model.fit(train_x, train_y, test_x, test_y, batch_size=20, epochs=4, callbacks=[eval_callback, tf_board_callback])
    model.evaluate(test_x, test_y)
    model.save('./model_8')

    # 预测结果
    df_out = pd.DataFrame(columns=['原文', '肿瘤原发部位', '原发病灶大小', '转移部位'])
    loaded_model = kashgari.utils.load_model('model_8')
    df = pd.read_excel("./data/test_no_ner.xlsx")
    for index, row in df.iterrows():
        data = row['原文']
        ''' Word2Vec '''
        Y_str, S_str, Z_str = predi_output(data, 'W2V', loaded_model)
        ''' Bert '''
        #Y_str, S_str, Z_str = predi_output(data, 'Bert', loaded_model)

        print(index, Y_str, S_str, Z_str)

        df_out = df_out.append({'原文': data, '肿瘤原发部位': Y_str, '原发病灶大小': S_str, '转移部位': Z_str}, ignore_index=True)