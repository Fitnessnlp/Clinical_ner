# 中文病历命名实体识别
    NER模型使用BiLSTM+CRF和BERT+BiLSTM+CRF两种，kashgari框架实现模型
    词向量Word2Vec用genism训练，利用1000条各个场景的非标注数据训练词向量  
    跨场景迁移用的是sklearn_crfsuite，基于CRF算法

## 配置：
>> tensorflow = 1.15  
>> kashgari = 1.1.1

## 代码：
    1. bare_bilstmcrf.py使用随机向量作为初始词向量，相关的数据预处理代码在Bare_BiLSTM_CRF文件夹  
    2. main_model.py模型训练与预测，采用word2Vec词嵌入和BERT词向量两种方式  
    3. transfer_target.py场景迁移部分  
    
  
## 模型结果
!(https://github.com/Fitnessnlp/Clinical_ner/blob/master/Bare_BiLSTM_CRF/1610012131(1).png) 
