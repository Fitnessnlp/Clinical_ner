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
1.bare_bilstm_crf  
![bare_bilstm+crf](https://github.com/Fitnessnlp/Clinical_ner/blob/master/Bare_BiLSTM_CRF/1610012131(1).png)   
2. word2vec:

            precision    recall  f1-score   support  
        TRA     0.6083    0.4193    0.4964       663  
        ORI     0.3327    0.2696    0.2978       638  
        SIZ     0.1579    0.1111    0.1304        27    
        micro avg     0.4516    0.3411    0.3887      1328  
        macro avg     0.4667    0.3411    0.3936      1328  

3.BERT:

                precision    recall  f1-score   support  
            TRA     0.2904    0.2635    0.2763       873  
            SIZ     0.1600    0.0792    0.1060       101  
            ORI     0.5845    0.2482    0.3484       822  
            
            micro avg     0.3711    0.2461    0.2959      1796  
            macro avg     0.4177    0.2461    0.2997      1796

