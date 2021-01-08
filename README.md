# 中文病历命名实体识别

## 目录结构：
├── Bare_BiLSTM_CRF  
│ ├──  bilstm_crf  
│ │ ├── model.py  
│ ├──  checkpoints  
│ │ ├── lstm_crf.pth  
│ ├──  data  
│ ├──  venv  
│ ├──  bilstm_crf_argparse.py  
│ ├──  consta.py  
│ ├──  data_loader.py  
│ ├──  processing_data.py  
│ ├──  tag2id.py  
├── data  
├── logs  
├── README.md  
├── bare_bilstm.py  
├── config.py  
├── download.txt  
├── main_model.py   
├── transfer_target.py  
├── word2vec.model  


## 说明：
    1. Bare_BiLSTM_CRF文件夹中，bilstm_crf_argparse.py是使用argparse设置模型参数；data_loader.py创建torch的DataLoader；processing_data.py是把病历文本用BIO进行标注，生成NER模型训练测试所需的数据格式；tag2id.py生成病历文本的语料库；
    2. bare_bilstmcrf.py使用随机向量作为初始词向量，训练并评估模型效果，模型及数据在Bare_BiLSTM_CRF文件夹
    3. 下载chinese_L-12_H-768_A-12，解压到一级目录（详见download.txt），他是一个BERT预训练模型，可以在这个预训练模型上面进行fine-tune
    4. main_model.py模型训练与预测，采用word2Vec词嵌入和BERT词向量两种方式；其中词向量Word2Vec用genism训练，利用1000条各个场景的非标注数据训练词向量；BERT embedding采用 chinese_L-12_H-768_A-12
    5. transfer_target.py场景迁移部分，跨场景迁移用的是sklearn_crfsuite，基于CRF算法

## 配置：
>> tensorflow = 1.15  
>> kashgari = 1.1.1  
>> Pytorch >= 1.0  

  
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

