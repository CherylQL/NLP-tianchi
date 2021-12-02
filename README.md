## NLP 赛题学习赛题介绍
https://tianchi.aliyun.com/competition/entrance/531810/introduction

赛题以匿名处理后的新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出 14 个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。

赛题数据由以下几个部分构成：训练集 20w 条样本，测试集 A 包括 5w 条样本，测试集 B 包括 5w 条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。评价标准为类别 f1_score 的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。

处理后的赛题训练数据如下：

label| text
---|---
6|57 44 66 56 2 3 3 37 5 41 9 57 44 47 45 33 13 63 58 31 17 47 0 1 1 69 26 60 62 15 21 12 49 18 38 20 50 23 57 44 45 33 25 28 47 22 52 35 30 14 24 69 54 7 48 19 11 51 16 43 26 34 53 27 64 8 4 42 36 46 65 69 29 39 15 37 57 44 45 33 69 54 7 25 40 35 30 66 56 47 55 69 61 10 60 42 36 46 65 37 5 41 32 67 6 59 47 0 1 1 68

在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

## 任务汇总
任务名称|难度|所需技能
---|---|---
报名比赛，下载比赛数据集并完成读取|低、1|Pandas
|对数据集字符进行可视化，统计标签和字符分布|中、2|Pandas
使用 TFIDF提取文本特征|中、2|Sklearn
使用 TFIDF 特征 和 线性模型完成训练和预测|中、2|Sklearn
使用 TFIDF 特征 和 XGBoost 完成训练和预测|中、2|Sklearn、XGBoost
学会训练 FastText、Word2Vec 词向量|中、2|FastText、gensim
使用 Word2Vec 词向量，搭建 TextCNN 模型训练预测|高、3|Pytorch、Keras使用 Word2Vec 词向量，搭建 BILSTM 模型训练预测|高、3|Pytorch、Keras学会 Bert 基础，transformer 库基础使用|中、2|Pytorch、transformer
使用 Bert在比赛数据集中完成预训练|高、3|Pytorch、transformer
使用 Bert 在比赛数据集上完成微调|高、3|Pytorch、transformer


### 任务1：报名比赛，下载比赛数据集并完成读取
- [x] 登录天池，可使用淘宝 & 支付宝登录，https://account.aliyun.com/login/login.htm
- [x] 下载比赛数据集，https://tianchi.aliyun.com/competition/entrance/531810/introduction读取比赛数据集，
- [x] 读取代码参考如下


    Pythonimport pandas as pd
    train_df = pd.read_csv('train_set.csv', sep='\t', nrows=100)
    train_df['word'] = train_df['text'].apply(lambda x: len(x.split(' ')))

### 任务2：对数据集字符进行可视化，统计标签和字符分布
- [x] 统计数据集中所有句子所包含字符的平均个数
- [x] 统计数据集中不同类别下句子平均字符的个数
- [x] 统计数据集中类别分布的规律
- [x] 统计数据集中不同类别下句子中最常见的5个字符

### 任务3：使用 TFIDF 提取文本特征

- [x] 学习TFIDF的原理
- [x] 学会使用CountVectorizer
- [x] 学会使用TfidfVectorizer

### 任务4：使用 TFIDF 特征 和 线性模型完成训练和预测

- [x] 使用TFIDF提取训练集和测试集特征
- [x] 使用线性模型（LR等）完成模型的训练和预测（10月11日）

### 任务5：使用 TFIDF 特征 和 XGBoost完成训练和预测

- [x] 使用TFIDF提取训练集和测试集特征
- [x] 使用XGBoost完成模型的训练和预测

### 任务6：学会训练 FastText、Word2Vec 词向量

- [ ] 学习词向量的基础原理和优点
- [ ] 学会训练FastText词向量
- [ ] 学会训练Word2Vec词向量

### 任务7：使用 Word2Vec 词向量，搭建 TextCNN 模型训练预测

- [ ] 学习TextCNN网络模型结构
- [ ] 学习深度学习中Embeeding层使用
- [ ] 使用深度学习框架（推荐Pytorch）搭建TextCNN，完成训练和预测

### 任务8：使用 Word2Vec 词向量，搭建 BILSTM 模型训练预测

- [ ] 学习BILSTM网络模型结构
- [ ] 使用深度学习框架（推荐Pytorch）搭建BILSTM，完成训练和预测

### 任务9：学会 Bert 基础，transformer 库基础

- [ ] 使用学习Bert基础和使用
- [ ] 学习transformer库的基础使用

### 任务10：使用 Bert 在比赛数据集中完成预训练
- [ ] 学习Bert的pretrain任务
- [ ] 使用Bert在比赛数据集中完成预训练

### 任务11：使用 Bert 在比赛数据集上完成微调
- [ ] 学会Bert的finetune任务
- [ ] 学习 Bert 在比赛数据集上完成微调