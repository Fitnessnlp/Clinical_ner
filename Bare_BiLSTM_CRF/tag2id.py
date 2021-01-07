#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/19 11:53
# @Author: chenwei
# @File  : tag2id.py

import numpy as np
import pickle
import os

tag2label = {
    "O":0,
    "B-ORI":1, "I-ORI":2,
    "B-SIZ":3, "I-SIZ":4,
    "B-TRA":5, "I-TRA":6,
    "START": 7, "STOP": 8
}

def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))
    return word2id

def read_corpus(train_data, is_train=True):
    sents = []
    labels = []
    data = []

    if not is_train:
        word2id = read_dictionary('./data/vocab.pkl')
    else:
        word2id = {}

    for line_co, line_lb in train_data:
        line_co = ' '.join(line_co)
        line_lb = ' '.join(line_lb)
        sent_ = line_co.strip().split()
        tag_ = line_lb.strip().split()
        data.append((sent_, tag_))

        sentence_id = []

        for word in sent_:
            if is_train:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = len(word2id) + 1
                sentence_id.append(word2id[word])  # sentence to id
            else:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word = '<UNK>'
                sentence_id.append(word2id[word])  # sentence to id

        label_ = []
        for tag in tag_:
            label = tag2label[tag]
            label_.append(label)

        sents.append(sentence_id)
        labels.append(label_)

    if is_train:
        word2id['<UNK>'] = len(word2id) + 1
        word2id['<PAD>'] = 0
        print('vocabulary length:', len(word2id))
        with open('./data/vocab.pkl', 'wb') as fw:
            pickle.dump(word2id, fw)

    return sents, labels, len(word2id), data

#if __name__ == '__main__':
#    npy_file = 'C:/Users/chenwei/PycharmProjects/pythonProject1/data/train.npy'
#    train_data = np.load(npy_file, allow_pickle=True)
#    sents, labels, t, data = read_corpus(train_data, is_train=True)



