#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/19 12:04
# @Author: chenwei
# @File  : main.py

import torch
import numpy as np
from Bare_BiLSTM_CRF.data_loader import DataLoader
from Bare_BiLSTM_CRF.tag2id import read_corpus, tag2label
from Bare_BiLSTM_CRF.bilstm_crf_argparse import get_argparse
from Bare_BiLSTM_CRF.bilstm_crf import model
import os

# 运行 .parse_args()
args = get_argparse().parse_args()
torch.manual_seed(args.seed)
args.use_cuda = True

# processing raw data
data_train = np.load('./Bare_BiLSTM_CRF/data/train.npy', allow_pickle=True)
data_test = np.load('./Bare_BiLSTM_CRF/data/test.npy', allow_pickle=True)
sents_train, labels_train, args.word_size, _ = read_corpus(data_train, is_train=True)
sents_test, labels_test, _, data_origin = read_corpus(data_test, is_train=False)
args.label_size = len(tag2label)

train_data = DataLoader(sents_train, labels_train, cuda=args.use_cuda, batch_size=args.batch_size)
test_data = DataLoader(sents_test, labels_test, cuda=args.use_cuda, shuffle=False, evaluation=True, batch_size=args.batch_size)

# 模型参数导入
bilstm_crf = model.Model(args)
if args.use_cuda:
    bilstm_crf = bilstm_crf.cuda()

optimizer = torch.optim.Adam(bilstm_crf.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.l2)

def train():
    bilstm_crf.train()
    total_loss = 0
    for word, label, seq_lengths, _  in train_data:
        optimizer.zero_grad()
        loss, _ = bilstm_crf(word, label, seq_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
    return total_loss / train_data._stop_step

def PRF(ori_lb, pred_lb, t_pred):
    '''
    ori_lb : 原始数据里的正样本
    pred_lb: 预测为正的样本
    t_pred : 正确预测为正的样本
    '''
    if pred_lb != 0:
        precision = float(float(t_pred)/float(pred_lb))
    else:
        precision = 0
    if ori_lb != 0:
        recall = float(float(t_pred)/float(ori_lb))
    else:
        recall = 0
    if precision + recall != 0:
        F_value = float(precision * recall * 2.0) / (precision + recall)
    else:
        F_value = 0
    #return precision, recall, F_value
    return F_value

'''获取precision, recall, F1'''
def get_P_R(label_list):
    # 原始样本的正样本
    T_ORI, T_SIZ, T_TRA = 0, 0, 0
    # 预测为正的样本
    P_ORI, P_SIZ, P_TRA = 0, 0, 0
    # 正确预测为正的正样本
    PT_ORI, PT_SIZ, PT_TRA = 0, 0, 0
    label2tag1 = {}
    for tag, lb in tag2label.items():
        label2tag1[lb] = tag
    for label_, (sent, tag) in zip(label_list, data_origin):
        tag_ = [label2tag1[label__] for label__ in label_]
        sent_len = len(sent)
        i = 0
        while i < sent_len:
            if tag[i] == 'B-ORI':
                ind = i + 1
                while ind < sent_len and tag[ind] == 'I-ORI':
                    ind += 1
                if tag_[i:ind] == tag[i:ind]:
                    PT_ORI += 1
                T_ORI += 1
                i = ind
            elif tag[i] == 'B-SIZ':
                ind = i + 1
                while ind < sent_len and tag[ind] == 'I-SIZ':
                    ind += 1
                if tag_[i:ind] == tag[i:ind]:
                    PT_SIZ += 1
                T_SIZ += 1
                i = ind
            elif tag[i] == 'B-TRA':
                ind = i + 1
                while ind < sent_len and tag[ind] == 'I-TRA':
                    ind += 1
                if tag_[i:ind] == tag[i:ind]:
                    PT_TRA += 1
                T_TRA += 1
                i = ind
            else:
                i += 1
        j = 0
        while j < sent_len:
            if tag_[j] == 'B-ORI':
                temp = j + 1
                while tag_[temp] == 'I-ORI':
                    temp += 1
                P_ORI += 1
                j = temp
            elif tag_[j] == 'B-SIZ':
                temp = j + 1
                while tag_[temp] == 'I-SIZ':
                    temp += 1
                P_SIZ += 1
                j = temp
            elif tag_[j] == 'B-TRA':
                temp = j + 1
                while tag_[temp] == 'I-TRA':
                    temp += 1
                P_TRA += 1
                j = temp
            else:
                j += 1
    # compute the Precision, Recall and F1 score
    #preci_ORI, recal_ORI, F_ORI = PRF(T_ORI, P_ORI, PT_ORI)
    #preci_SIZ, recal_SIZ, F_SIZ = PRF(T_SIZ, P_SIZ, PT_SIZ)
    #preci_TRA, recal_TRA, F_TRA = PRF(T_TRA, P_TRA, PT_TRA)
    F_ORI = PRF(T_ORI, P_ORI, PT_ORI)
    F_SIZ = PRF(T_SIZ, P_SIZ, PT_SIZ)
    F_TRA = PRF(T_TRA, P_TRA, PT_TRA)
    return F_ORI, F_SIZ, F_TRA

def evaluate(epoch):
    bilstm_crf.eval()
    eval_loss = 0

    model_predict = []
    sent_res = []

    label2tag = {}
    for tag, lb in tag2label.items():
        label2tag[lb] = tag if lb != 0 else lb

    label_list = [] 

    for word, label, seq_lengths, unsort_idx in test_data:
        loss, _ = bilstm_crf(word, label, seq_lengths)
        pred = bilstm_crf.predict(word, seq_lengths)
        pred = pred[unsort_idx]
        seq_lengths = seq_lengths[unsort_idx]

        for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
            pred_ = list(pred[i][:seq_len].cpu().numpy())
            label_list.append(pred_)

        eval_loss += loss.detach().item()

    # 获取F1 value
    F_ORI, F_SIZ, F_TRA = get_P_R(label_list)
    print('| F1 - 原发部位 {:2.6f} | F1 - 肿瘤大小: {:2.6f} | F1 - 转移部位 {:2.6f}'.format(
        F_ORI, F_SIZ, F_TRA
    ))

    for label_, (sent, tag) in zip(label_list, data_origin):
        # 
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if len(label_) != len(sent):
            # print(sent)
            print(len(sent))
            print(len(label_))
            # print(tag)
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)



    return eval_loss / test_data._stop_step


import time

train_loss = []
if args.mode == 'train':
    best_acc = None
    total_start_time = time.time()

    print('-' * 90)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss * 1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
            epoch, time.time() - epoch_start_time, loss))
        eval_loss = evaluate(epoch)
        torch.save(bilstm_crf.state_dict(), args.save)



