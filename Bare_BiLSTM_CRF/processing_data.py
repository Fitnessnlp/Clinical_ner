#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/17 16:38
# @Author: chenwei
# @File  : processing_data.py
import numpy as np
import pandas as pd
import re
import xlsxwriter
import logging

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def extract_entity(text, type_ent):
    # 如果text为Nan
    if pd.isna(text):
        return  [(None, type_ent)]
    return [(s, type_ent) for s in text.split(',') if s != '']

def text_filter(text):
    # 去掉无意义的空格，小方块
    # text = re.sub(' ','',text)

    target = [ ';', '；']

    for i in target:
        text = text.replace(i, '。')

    return text

#    def text_split(self, text):
    #text = re.split(r'[;,。\s]', text)
#        text = self.text_filter(text)
#        text1 = re.split(r',', text)
#        return text1
def text_split(text):
    text = str(text)
    # 连续多个空格，合并为一个
    text = re.sub(r'\s+',' ', text)
    # 去除空格
    text = re.sub(' ', '', text)
    #text = text_filter(text)
    text1 = re.split(r'。', text)

    return text1

def get_all_entity(data):
    # 逐行遍历data
    BIO_result = []
    for _, i in data.iterrows():
        # print(i['原发病灶大小'])
        temp = []
        temp.extend(extract_entity(i['肿瘤原发部位'], 'ORI'))
        temp.extend(extract_entity(i['原发病灶大小'], 'SIZ'))
        temp.extend(extract_entity(i['转移部位'], 'TRA'))
        temp = [j for j in temp if j[0] is not None]

        temp = sorted(temp, key=lambda x:len(x[0]))

        BIO_result.append(temp)

    return BIO_result

def get_position(text, entities):
    #
    test_posit = ['O'] * len(text)

    for entity in entities:
        al = re.finditer(re.escape(entity[0]), text)
        for i in al:
            position = i.span()
            test_posit[position[0]:position[1]] = ['B-' + entity[1]] + ['I-' + entity[1]] * (len(entity[0]) - 1)
    return (list(text), test_posit)

def get_bio_pos(data:pd.DataFrame):
    #data = self.load_data()
    data['实体'] = get_all_entity(data)
    po_result = []
    for _, line in data.iterrows():
        yuanwen = line['原文']
        enti = line['实体']
        yuanwen1 = text_split(yuanwen)
        for t in yuanwen1:
            if t == '':
                continue
            text, test_posit = get_position(t, enti)
            po_result.append((text, test_posit))

    return po_result

if __name__ == '__main__':

    data1 = pd.read_excel('C:/Users/Caesar/Desktop/NER_tasks/NER-ccks2019--master/Bare_BiLSTM_CRF/data/training_part1.xlsx')
    data2 = pd.read_excel('C:/Users/Caesar/Desktop/NER_tasks/NER-ccks2019--master/Bare_BiLSTM_CRF/data/training_part2.xlsx')
    data = pd.concat((data1, data2), axis=0, ignore_index=True)
    test_data = pd.read_excel('C:/Users/Caesar/Desktop/NER_tasks/NER-ccks2019--master/Bare_BiLSTM_CRF/data/test.xlsx')
    #result1 = get_bio_pos(data1)
    #result2 = get_bio_pos(data2)
    result = get_bio_pos(data)
    result = np.array(result)
    #print(result)
    result_test = get_bio_pos(test_data)
    result_test = np.array(result_test)

    np.save('C:/Users/Caesar/Desktop/NER_tasks/NER-ccks2019--master/Bare_BiLSTM_CRF/data/train.npy', result)
    np.save('C:/Users/Caesar/Desktop/NER_tasks/NER-ccks2019--master/Bare_BiLSTM_CRF/data/test.npy', result_test)




