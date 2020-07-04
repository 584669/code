import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

#
# 参考了onion选手的数据求差思路进行数据扩充
# 数据扩充后使用相对距离进行数据筛选
# 由于xgb训练非常慢，为了提高训练效率使用双lgb方式
#https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.12.54587b9dUSx0my&postId=54342
##########.......................
train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
A = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
A['A25'] = A['A25'].fillna(70)
Aans = pd.read_csv('./data/jinnan_round1_ansA_20190125.csv', header=None)
A["收率"] = Aans[1]
B = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding='gb18030')
Bans = pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', header=None)
B["收率"] = Bans[1]
C = pd.read_csv('./data/jinnan_round1_test_20190201.csv', encoding='gb18030')
Cans = pd.read_csv('./data/jinnan_round1_ans_20190201.csv', header=None)
C["收率"] = Cans[1]
train = pd.concat([train, A, B, C], axis=0, ignore_index=True)

test = pd.read_csv('./data/FuSai.csv', encoding='gb18030')  # FuSai
create_csv = pd.read_csv('./data/FuSai.csv', encoding='gb18030')
###############
# 筛选原始数据
train = train[train['收率'] > 0.86]
train.loc[train['B14'] == 40, 'B14'] = 400
train = train[train['B14'] >= 400]
#
target = train['收率']
train.loc[train['A25'] == '1900/3/10 0:00', 'A25'] = train['A25'].value_counts().values[0]
train['A25'] = train['A25'].astype(int)
#
test['A25'] = test['A25'].astype(int)
train.loc[train['B14'] == 40, 'B14'] = 400

test_select = {}
for v in [280, 385, 390, 785]:
    test_select[v] = test[test['B14'] == v]['样本id'].index

del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)


def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0
    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600
    return tm


for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f)


def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1
    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1
    return tm


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)

data['样本id'] = data['样本id'].apply(lambda x: x.split('_')[1])
data['样本id'] = data['样本id'].astype(int)
# 重新对每一个数据编号
if os.path.exists('./csei.csv'):
    data['样本id'] = pd.read_csv('./csei.csv', encoding='gb18030').values
else:
    seipd = pd.DataFrame()
    sei = np.arange(len(data))
    np.random.shuffle(sei)
    seipd['样本id'] = sei
    seipd.to_csv('./csei.csv', index=False)
    data['样本id'] = pd.read_csv('./csei.csv', encoding='gb18030').values
#
categorical_columns = [f for f in data.columns if f not in ['样本id', '收率']]
data_ = pd.DataFrame()
# 数据归一化
for f in categorical_columns:
    data_[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    maxn = data_[f].values.max()
    data_[f] = data_[f] / maxn
# 原点的相对距离
dis_n = np.zeros((len(data_)))
for n in range(len(data_)):
    dis_n[n] = np.sqrt(np.sum(np.square(data_.values[n])))

data['dis'] = dis_n
#
data['b14/a1_a3_a4_a19_b1_b12'] = 100 * data['B14'] / (
            data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12'])
#
train = data[:train.shape[0]]
test = data[train.shape[0]:]

train['target'] = list(target)

new_train = train.copy()
new_train = new_train.sort_values(['样本id'], ascending=True)

train_copy = train.copy()
train_copy = train_copy.sort_values(['样本id'], ascending=True)
#
train_len = len(new_train)
new_train = pd.concat([new_train, train_copy])
#
new_test = test.copy()
new_test = pd.concat([new_test, new_train])
#
diff_train = pd.DataFrame()
ids = list(train_copy['样本id'].values)
idst = test.sort_values(['样本id'], ascending=True)
#
from tqdm import tqdm

num = train_len
# 构造新的训练集
diff_trainp = './diff_train.csv'
diff_testp = './diff_test.csv'

if os.path.exists(diff_trainp):
    diff_train = pd.read_csv(diff_trainp, encoding='gb18030')
else:
    for i in tqdm(range(1, num)):
        #
        diff_tmp = new_train.diff(-i)
        diff_tmp = diff_tmp[:train_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        #
        diff_tmp['样本id'] = ids

        diff_tmp = diff_tmp[diff_tmp['dis_difference'] <= 0.23]
        diff_tmp = diff_tmp[diff_tmp['dis_difference'] >= -0.23]

        diff_train = pd.concat([diff_train, diff_tmp])

    diff_train.to_csv(diff_trainp, index=False)

#
diff_test = pd.DataFrame()
ids_test = list(test['样本id'].values)
test_len = len(test)

if os.path.exists(diff_testp):
    diff_test = pd.read_csv(diff_testp, encoding='gb18030')
else:
    for i in tqdm(range(test_len, test_len + num)):
        #
        diff_tmp = new_test.diff(-i)
        diff_tmp = diff_tmp[:test_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        #
        diff_tmp['样本id'] = ids_test

        diff_tmp = diff_tmp[diff_tmp['dis_difference'] <= 0.23]
        diff_tmp = diff_tmp[diff_tmp['dis_difference'] >= -0.23]

        diff_test = pd.concat([diff_test, diff_tmp])

    diff_test = diff_test[diff_train.columns]
    diff_test.to_csv(diff_testp, index=False)
#
train_target = train['target']
train.drop(['target'], axis=1, inplace=True)
#
diff_train = pd.merge(diff_train, train, how='left', on='样本id')
diff_test = pd.merge(diff_test, test, how='left', on='样本id')
#
target = diff_train['target_difference']
diff_train.drop(['target_difference'], axis=1, inplace=True)
diff_test.drop(['target_difference'], axis=1, inplace=True)

X_train = diff_train.copy()
y_train = target.copy()
X_test = diff_test.copy()
#
diff_trainrsv = diff_train.copy()
diff_testrsv = diff_test.copy()
#
X_train['b8/b6'] = X_train['B8'] / X_train['B6']
X_test['b8/b6'] = X_test['B8'] / X_test['B6']
#
# 去除ID相关
X_train.drop(['样本id'], axis=1, inplace=True)
X_test.drop(['样本id'], axis=1, inplace=True)
X_train.drop(['样本id_difference'], axis=1, inplace=True)
X_test.drop(['样本id_difference'], axis=1, inplace=True)
#########双lgb..分别使用mse和mae的方式######################
param = {'num_leaves': 81,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l2": 0.1,
         'num_thread': 4,
         "verbosity": -1}

param2 = {'num_leaves': 81,
          'min_data_in_leaf': 20,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "lambda_l2": 0.1,
          'num_thread': 4,
          "verbosity": -1}
#
folds = KFold(n_splits=3, shuffle=True, random_state=2019)
oof_lgb = np.zeros(len(diff_train))
predictions_lgb = np.zeros(len(diff_test))

for n_ in range(2):
    diff_train = diff_trainrsv
    diff_test = diff_testrsv
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        dev = X_train.iloc[trn_idx]
        val = X_train.iloc[val_idx]

        trn_data = lgb.Dataset(dev, y_train.iloc[trn_idx])
        val_data = lgb.Dataset(val, y_train.iloc[val_idx])

        num_round = 500
        if n_ == 0:
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=5,
                            early_stopping_rounds=100)
        else:
            clf = lgb.train(param2, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=5,
                            early_stopping_rounds=100)
        oof_lgb[val_idx] = clf.predict(val, num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    #
    diff_train['compare_id'] = diff_train['样本id'] - diff_train['样本id_difference']
    train['compare_id'] = train['样本id']
    train['compare_target'] = list(train_target)
    #
    diff_train = pd.merge(diff_train, train[['compare_id', 'compare_target']], how='left', on='compare_id')
    # print(diff_train.columns.values)
    diff_train['pre_target_diff'] = oof_lgb
    diff_train['pre_target'] = diff_train['pre_target_diff'] + diff_train['compare_target']
    #
    mean_result = diff_train.groupby('样本id')['pre_target'].mean().reset_index(name='pre_target_mean')
    true_result = train[['样本id', 'compare_target']]
    mean_result = pd.merge(mean_result, true_result, how='left', on='样本id')
    # print(mean_result)
    print("CV2 score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
    #
    print("CV3 score: {:<8.8f}".format(
        mean_squared_error(mean_result['pre_target_mean'].values, mean_result['compare_target'].values)))
    ################################################################################################
    #
    diff_test['compare_id'] = diff_test['样本id'] - diff_test['样本id_difference']
    diff_test = pd.merge(diff_test, train[['compare_id', 'compare_target']], how='left', on='compare_id')
    diff_test['pre_target_diff'] = predictions_lgb
    diff_test['pre_target'] = diff_test['pre_target_diff'] + diff_test['compare_target']
    ##
    mean_result_test = diff_test.groupby(diff_test['样本id'], sort=False)['pre_target'].mean().reset_index(
        name='pre_target_mean')

    if n_ == 0:
        test1 = pd.merge(test, mean_result_test, how='left', on='样本id')
    else:
        test2 = pd.merge(test, mean_result_test, how='left', on='样本id')

sub_df = pd.DataFrame()
sub_df[0] = create_csv['样本id']
sub_df[1] = (test1['pre_target_mean'] + test2['pre_target_mean']) / 2
sub_df[1] = sub_df[1].apply(lambda x: round(x, 3))

sub_df.to_csv('./submit_FuSai.csv', index=False, header=False)