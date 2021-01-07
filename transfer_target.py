#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/12/28 20:47
# @Author: chenwei
# @File  : transfer_target.py

import numpy as np
import re
import pandas as pd
import xlsxwriter
import sklearn_crfsuite
from Bare_BiLSTM_CRF.tag2id import tag2label


'''跨场景迁移；目标场景迁移到非目标场景'''
CRF = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=1e-3, max_iterations=100, all_possible_transitions=True)

def text2pos(text,ners):
    text_list = list(text)
    text_ner_list = ['O'] * len(text_list)
    for ner in ners:
        ms = re.finditer(re.escape(ner[0]), text)
        for m in ms:
            ner_type = ner[1]
            postion = m.span()
            text_ner_list[postion[0]:postion[1]] = ['B-' + ner_type] + ['I-' + ner_type] * (len(ner[0]) - 1)

    return (text_list,text_ner_list)


def text_split(text):
    text = str(text)
    text = re.split('。|；|;| ,|，|；|,|\t| ', text)
    return text


def get_ners_postion(data) :
    ners = get_ners(data)
    data['ners'] = ners
    result = []
    for index_,row in data.iterrows():
        text = row['原文']
        ners = row['ners']
        texts = text_split(text)
        for text in texts:
            if text == '':
                continue
            text_list,pos_list = text2pos(text,ners)
            result.append((text_list,pos_list))

    return result


#一个item中所有的实体
def get_ner(s,ner):
    if s is None or pd.isna(s):
        return [(None,ner)]

    return [(i,ner) for i in s.split(',') if i!= '']


#得到itmes的所有实体
def get_ners(data:pd.DataFrame):
    result = []
    for _,i in data.iterrows():
        result_i =[]
        result_i.extend(get_ner(i['肿瘤原发部位'],'ORI'))
        result_i.extend(get_ner(i['原发病灶大小'],'SIZ'))
        result_i.extend(get_ner(i['转移部位'],'TRA'))
        result_i = [j for j in result_i if j[0] is not None]
        #排序
        result_i = sorted(result_i,key=lambda x:len(x[0]))
        result.append(result_i)

    return result


def tag2num(labels):
    nums = []

    for row in labels:
        temp_list = []
        for j in row[0]:
            temp_list.append(tag2label.get(j))
        temp_list = np.array(temp_list)
        nums.append(temp_list)
    nums = np.array(nums).reshape((len(nums), 1))
    return nums

def change2list(yangben):
    temp_yangben = yangben.tolist()
    result = []
    for i in temp_yangben:
        result.append(i[0])
    #result = np.array(result).reshape((len(result), 1))
    return result

def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')

def train_classify(trains_data, trains_label, test_data, P):

    # clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    CRF.fit(trains_data, trains_label)
    # BiLSTM_CRF_Model.evaluate()
    return [CRF.predict(test_data), CRF.predict_marginals(test_data)]

def output_train_excel(trains_data, trains_label, sheet1, row_in_BLSTMCRF):
    # 原发部位
    primary_site = []
    # 转移部位
    transfer_area = []
    # 病灶大小
    size = []

    primary_site_start_pos = []
    primary_site_end_pos = []

    transfer_area_start_pos = []
    transfer_area_end_pos = []

    size_start_pos = []
    size_end_pos = []

    # 得到原发部位的位置
    for start in range(len(trains_label)):
        if "B-ORI" == trains_label[start]:
            primary_site_start_pos.append(start)
    for end in primary_site_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "I-ORI":
            end = end + 1
        primary_site_end_pos.append(end)


    # 得到转移部位的位置
    for start in range(len(trains_label)):
        if "B-TRA" == trains_label[start]:
            transfer_area_start_pos.append(start)
    for end in transfer_area_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "I-TRA":
            end = end + 1
        transfer_area_end_pos.append(end)


    # 得到病灶大小的位置
    for start in range(len(trains_label)):
        if "B-SIZ" == trains_label[start]:
            size_start_pos.append(start)
    for end in size_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "I-SIZ":
            end = end + 1
        size_end_pos.append(end)

    # 得到在原文中的原发部位的文字
    for i in range(len(primary_site_start_pos)):
        temp = ""
        t1 = trains_data[primary_site_start_pos[i]: primary_site_end_pos[i] + 1]
        temp = temp.join(t1)
        primary_site.append(temp)

    # 去除重复元素
    l2 = {}.fromkeys(primary_site).keys()
    primary_site = l2

    # 得到在原文中转移部位的文字
    for i in range(len(transfer_area_start_pos)):
        temp = ""
        t1 = trains_data[transfer_area_start_pos[i]: transfer_area_end_pos[i] + 1]
        temp = temp.join(t1)
        transfer_area.append(temp)

    # 去出重复元素
    l2 = {}.fromkeys(transfer_area).keys()
    transfer_area = l2

    # 得到在原文中原发病灶大小的文字
    for i in range(len(size_start_pos)):
        temp = ""
        t1 = trains_data[size_start_pos[i]: size_end_pos[i] + 1]
        temp = temp.join(t1)
        size.append(temp)

    # 出去重复元素
    l2 = {}.fromkeys(size).keys()
    size = l2

    # 写出原文到excel
    str = ""
    str = str.join(trains_data)
    sheet1.write(row_in_BLSTMCRF, 0, str)

    # 写出原发部位到excel
    out_str = ""
    index = 0
    for text in primary_site:
        index = index + 1
        out_str = out_str + text
        if index < len(primary_site):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 1, out_str)

    # 写出原发病灶大小到excel
    out_str = ""
    index = 0
    for text in size:
        index = index + 1
        out_str = out_str + text
        if index < len(size):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 2, out_str)

    # 写出转移部位到excel
    out_str = ""
    index = 0
    for text in transfer_area:
        index = index + 1
        out_str = out_str + text
        if index < len(transfer_area):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 3, out_str)


def transfer_scene(ori_data, tar_data, N):
    '''
    :param ori_data: 非目标域场景的数据
    :param tar_data: 100条目标场景的标注数据
    :param N: 迭代次数
    :return:
    '''
    ori_data = np.array(ori_data)
    tar_data = np.array(tar_data)
    # test = np.array(test)
    # 数据的原文
    ori_text, tar_text = ori_data[:, 0], tar_data[:, 0]
    # test_text = test[:, 0]
    ori_text = ori_text.reshape((ori_text.shape[0], 1))
    tar_text = tar_text.reshape((tar_text.shape[0], 1))
    # test_text = test_text.reshape((test_text.shape[0], 1))
    # 数据的标签
    ori_label, tar_label = ori_data[:, 1], tar_data[:, 1]
    # test_label = test[:, 1]
    ori_label = ori_label.reshape((ori_label.shape[0], 1))
    tar_label = tar_label.reshape((tar_label.shape[0], 1))
    # test_label = test_label.reshape((test_label.shape[0], 1))

    # 标签2id，转换成对应的数字
    ori_int_label = tag2num(ori_label)
    tar_int_label = tag2num(tar_label)
    # test_int_label = tag2num(test_label)

    # 数据，标签竖直方向拼接
    datas = np.vstack((tar_text, ori_text))
    labels = np.vstack((tar_label, ori_label))
    row_A = tar_text.shape[0]
    row_S = ori_text.shape[0]

    test_data = datas
    # test_data = np.vstack((trains_data, test_sample))

    # 初始化权重，为每一个list分配权重
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    beta = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))
    # 存储每一次迭代的标签和beta值
    beta_T = np.zeros([1, N])

    # 训练样本的数据与标签转换为[[], [], []]的形式
    trains_data = change2list(datas)
    trains_label = change2list(labels)

    # 测试样本也做同样的处理
    tests_data = change2list(test_data)

    P = weights
    for i in range(N):
        P = calculate_P(weights, trains_label)
        # temp返回值为 temp[0]是对test_data的预测标签     temp[1]是每个预测标签的概率
        temp = train_classify(trains_data, trains_label, test_data, P)
        temp_predict_list = temp[0]
        # 将CRF的预测值对应到相应的数字
        predict_list = []
        for k in temp_predict_list:
            list_temp = []
            for j in k:
                list_temp.append(tag2label.get(j))
            predict_list.append(list_temp)
        predict_list = np.array(([np.array(i) for i in predict_list]))
        predict_list = predict_list.reshape(predict_list.shape[0], 1)

        # 计算错误率  计算的是一个目标域list的整个错误率
        # 目标场景的预测情况
        error_rate = 0
        predict_probability = temp[1]
        for j in range(row_A):
            a_sample = tar_label[j, :][0]
            temp_dict_list = predict_probability[j]
            res = 0
            for k in range(len(temp_dict_list)):
                each_dict = temp_dict_list[k]
                max_probability_tuple = max(zip(each_dict.values(), each_dict.keys()))
                if max_probability_tuple[1] == a_sample[k]:
                    res = res
                else:
                    res = res + 1 - each_dict.get(a_sample[k])
            ## 错误率是当前一个list每个标签的错误概率相加之后 * 该标签的权重
            error_rate = error_rate + res * weights[j, :]
            if np.isnan(error_rate):
                print('出现nan值得地方是'+ str(j))
                break
        print('Error rate:', error_rate)

        beta_T[0, i] = np.abs(error_rate / (1 - error_rate))
        # 非目标场景样本权重调整
        for j in range(row_S):
            power = 0
            temp_array = np.abs(predict_list[row_A + j] - ori_int_label[j])
            # 类似于计算loss
            for y in temp_array:
                power = power + np.sum(y)

            if power == 0:
                power == power
            else:
                power = 0.5 / power
            # 由目标场景的错误率情况更新非目标场景的样本权重
            # 根据beta[0,i]与power的情况更新对应的权重
            if beta_T[0, i] > 1.0:
                weights[row_A + j] = weights[row_A + j] * np.power(beta_T[0, i], power)
            else:
                if power == 0:
                    weights[row_A + j] = weights[row_A + j] * np.power(beta_T[0, i], power)
                else:
                    power = 0.5 / power
                    weights[row_A + j] = weights[row_A + j] * np.power(beta_T[0, i], power)


        # 调整目标域样本权重
        # 目标场景中预测标签与真实标签的误差情况更新权重
        for j in range(row_A):
            power = 0

            temp_array = np.abs(predict_list[j] - tar_int_label[j])
            for y in temp_array:
                power = power + np.sum(y)
            if power == 0:
                power = power
            else:
                power = 1 / power

            weights[j] = weights[j] * np.power(beta, power)

        print("*"*100)

    total_A = np.sum(weights[0:row_A-1, :])
    P[0:row_A-1, :] = weights[0:row_A-1, :] / total_A
    total_S = np.sum(weights[row_A: row_A+row_S-1, :])
    P[row_A: row_A + row_S-1, :] = weights[row_A: row_A+row_S-1, :] / total_S

    # 100条目标场景和900条非目标场景输出到excel文件
    row_in_BLSTMCRF = 1
    work_book = xlsxwriter.Workbook('./data/BLSTMCRF_DATA1.xlsx')
    sheet1 = work_book.add_worksheet('sheet1')
    sheet1.write(0, 0, "原文")
    sheet1.write(0, 1, "肿瘤原发部位")
    sheet1.write(0, 2, "原发病灶大小")
    sheet1.write(0, 3, "转移部位")

    # 100条目标场景
    for i in range(row_A):
        sample_weight = weights[i, 0]
        repetition_times = sample_weight * (row_A + row_S)
        temp = int(np.floor(repetition_times))
        nres_list = ["B-ORI", "I-ORI", "B-SIZ", "I-SIZ", "B-TRA", "I-TRA"]
        how_many_nres_in_label = 0
        for j in nres_list:
            if j in trains_label[i]:
                how_many_nres_in_label = how_many_nres_in_label + 1
        if how_many_nres_in_label == 0:
            temp = temp - 5

        for k in range(temp):
            if temp < 1:
                break
            output_train_excel(trains_data[i], trains_label[i], sheet1, row_in_BLSTMCRF)
            row_in_BLSTMCRF = row_in_BLSTMCRF + 1




    # 900条非目标场景
    for i in range(row_S):

        sample_weight = weights[row_A + i, 0]
        repetition_times = sample_weight * (row_A + row_S)
        temp = int(np.ceil(repetition_times))
        if temp > 5:
            temp = 3
        for j in range(temp):
            if temp < 1:
                break
            output_train_excel(trains_data[row_A + i], trains_label[row_A + i], sheet1, row_in_BLSTMCRF)
            row_in_BLSTMCRF = row_in_BLSTMCRF + 1
    work_book.close()


if __name__ == "__main__":

    # datathree = pd.read_excel("./data/BLSTMCRF_DATA1.xlsx")
    data1 = pd.read_excel('./data/training_part1.xlsx')
    data2 = pd.read_excel('./data/training_part2.xlsx')
    test_data = pd.read_excel('./data/test.xlsx')
    result1 = get_ners_postion(data1)
    result2 = get_ners_postion(data2)
    #result1 = np.array(result1)
    #result2 = np.array(result2)
    #result_test = np.array(result_test)

    N = 5
    transfer_scene(result2, result1, N)