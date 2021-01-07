#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/20 16:37
# @Author: chenwei
# @File  : bilstm_crf_argparse.py
import argparse
'''
import argparse
创建 parser
向 parser 添加位置变量和可选变量
运行 .parse_args()
'''
def get_argparse():
    # 创建parser
    parser = argparse.ArgumentParser(description='LSTM_CRF')
    # 向 parser 添加位置变量和可选变量
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs for train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--use-cuda', action='store_true',
                        help='enables cuda')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use-crf', action='store_true',
                        help='use crf')

    parser.add_argument('--mode', type=str, default='train',
                        help='train mode or test mode')

    parser.add_argument('--save', type=str, default='./Bare_BiLSTM_CRF/checkpoints/lstm_crf.pth',
                        help='path to save the final model')
    parser.add_argument('--save-epoch', action='store_true',
                        help='save every epoch')
    parser.add_argument('--data', type=str, default='dataset',
                        help='location of the data corpus')

    parser.add_argument('--word-ebd-dim', type=int, default=100,
                        help='number of word embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout')
    parser.add_argument('--lstm-hsz', type=int, default=300,
                        help='BiLSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='biLSTM layer numbers')
    parser.add_argument('--l2', type=float, default=0.005,
                        help='l2 regularization')
    parser.add_argument('--clip', type=float, default=.5,
                        help='gradient clipping')
    #parser.add_argument('--result-path', type=str, default='C:/Users/Caesar/Desktop/NER_Chinese/result',
    #                    help='result-path')

    return parser