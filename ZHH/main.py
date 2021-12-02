import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
def main():
    data = pd.read_csv('train_set.csv', sep='\t', nrows=10000)
    dft = pd.DataFrame(data)

    dft['x'] = dft['text']
    dft['y'] = dft['label']

    xtrain, xvalid, ytrain, yvalid = train_test_split(dft.x.values, dft.y.values,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)

    ctv = CountVectorizer(
                        # min_df=3,
                        #   max_df=0.9,
                          ngram_range=(1, 2))

    # 使用Count Vectorizer来fit训练集和测试集（半监督学习）
    ctv.fit(list(xtrain) + list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xvalid)

    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_ctv.tocsc(), ytrain)
    predictions = clf.predict_proba(xvalid_ctv.tocsc())

    print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    #
    # tfv = NumberNormalizingVectorizer(
    #                                 min_df=3,
    #                                 #   max_df=0.9,
    #                                   max_features=None,
    #                                   ngram_range=(1, 2),
    #                                   use_idf=True,
    #                                   smooth_idf=True)
    #
    #
    # # 使用TF-IDF来fit训练集和测试集（半监督学习）
    # tfv.fit(list(xtrain) + list(xvalid))
    # xtrain_tfv = tfv.transform(xtrain)
    # xvalid_tfv = tfv.transform(xvalid)
    # #
    # # # 利用提取的TFIDF特征来fit一个简单的Logistic Regression
    # # clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
    # # clf.fit(xtrain_tfv, ytrain)
    # # predictions = clf.predict_proba(xvalid_tfv)
    # #
    # # print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    # clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
    #                         subsample=0.8, nthread=10, learning_rate=0.1)
    # clf.fit(xtrain_tfv.tocsc(), ytrain)
    # predictions = clf.predict_proba(xvalid_tfv.tocsc())
    #
    # print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


main()