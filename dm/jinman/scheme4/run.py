import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.54587b9dUSx0my&postId=54332
# In[2]:
def main(testFile=''):
    train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
    test = pd.read_csv('./data/'+testFile, encoding='gb18030')  # let data C as test


    testA = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
    ansA = pd.read_csv('./data/jinnan_round1_ansA_20190125.csv', encoding='gb18030', header=None)
    ansA.columns = ['id', '收率']
    trainA = pd.concat([testA, ansA[['收率']]], axis=1)

    # In[5]:

    testB = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding='gb18030')
    ansB = pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', encoding='gb18030', header=None)
    ansB.columns = ['id', '收率']
    trainB = pd.concat([testB, ansB[['收率']]], axis=1)

    # In[6]:

    testC = pd.read_csv('./data/jinnan_round1_test_20190201.csv', encoding='gb18030')
    ansC = pd.read_csv('./data/jinnan_round1_ans_20190201.csv', encoding='gb18030', header=None)
    ansC.columns = ['id', '收率']
    trainC = pd.concat([testC, ansC[['收率']]], axis=1)


    time_point_cols = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
    time_period_cols = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']

    meterial_cols = ['A1', 'A2', 'A3', 'A4', 'A19', 'A21', 'A22', 'A23', 'B1', 'B12', 'B13', 'B14']
    temperature_cols = ['A6', 'A8', 'A10', 'A12', 'A15', 'A17', 'A25', 'A27', 'B6', 'B8']
    PH_cols = ['B2', 'B3']
    Kpa_cols = ['A13', 'A18']

    # In[9]:

    train = pd.concat([train, trainA, trainB, trainC], ignore_index=True)

    # ## 特征工程

    good_cols = list(train.columns)
    for col in train.columns:
        rate = train[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > 0.9:
            good_cols.remove(col)
            print(col, rate)

    # 构造特征后再删除
    good_cols.append('A1')
    good_cols.append('A3')
    good_cols.append('A4')
    good_cols.append('A13')
    good_cols.append('A18')

    # 删除异常值  # 0.85-1
    train = train[train['收率'] >= 0.85]
    train['收率'][train['收率'] >= 1] = 1
    train = train[good_cols]
    good_cols.remove('收率')
    test = test[good_cols]


    train = train[(train.B14 >= 350) & (train.B14 <= 460)]


    test_ids = test['样本id']

    # 合并数据集
    target = train['收率']
    # del train['收率']
    data = pd.concat([train, test], axis=0, ignore_index=True)
    del data['样本id']

    # In[17]:

    data.replace('19:-20:05', '19:05-20:05', inplace=True)
    data.replace('15:00-1600', '15:00-16:00', inplace=True)

    data.B1.replace(3.5, 350, inplace=True)
    data.A25 = data.A25.replace('1900/3/10 0:00', data['A25'].value_counts().values[0]).astype(float)
    data.loc[data['B14'] == 40, 'B14'] = 400
    # B14  350-460

    # In[18]:

    for col in time_point_cols:
        print(col, pd.to_datetime(data[col], errors='coerce').dt.second.nunique(), end=' ')

    for col in time_point_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce').dt.hour + pd.to_datetime(data[col],
                                                                                        errors='coerce').dt.minute / 60

    for col in time_period_cols:
        data[col + '_start'] = data[col].apply(str).apply(lambda x: x.split('-')[0])
        data[col + '_end'] = data[col].apply(str).apply(lambda x: x.split('-')[1] if len(x.split('-')) > 1 else np.NaN)

    start_end_list = ['A20_start', 'A20_end', 'A28_start', 'A28_end', 'B4_start',
                      'B4_end', 'B9_start', 'B9_end', 'B10_start', 'B10_end', 'B11_start',
                      'B11_end']

    for col in start_end_list:
        data[col] = pd.to_datetime(data[col], errors='coerce').dt.hour + pd.to_datetime(data[col],
                                                                                        errors='coerce').dt.minute / 60

    ## 特征工程

    time_point_list = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A20_start', 'A20_end', 'A24', 'A26', 'A28_start',
                       'A28_end', 'B4_start',
                       'B4_end', 'B5', 'B7', 'B9_start', 'B9_end', 'B10_start', 'B10_end', 'B11_start', 'B11_end']

    # time period
    for c1, c2 in zip(time_point_list, time_point_list[1:]):
        data['%s_%s_time' % (c1, c2)] = (data[c2] - data[c1]).apply(lambda x: x + 24 if x < 0 else x)

    data['all_time'] = data['B11_end'] - data['A5']

    # In[19]:

    start_end_list = ['A20_start', 'A20_end', 'A28_start', 'A28_end', 'B4_start',
                      'B4_end', 'B9_start', 'B9_end', 'B10_start', 'B10_end', 'B11_start',
                      'B11_end']

    # In[20]:

    # temperature diff
    for c1, c2 in zip(temperature_cols, temperature_cols[1:]):
        data['%s_%s_tempDiff' % (c1, c2)] = data[c2] - data[c1]
    data['%s_%s_tempDiff' % ('A6', 'B8')] = data['B8'] - data['A6']

    data.drop(time_period_cols, inplace=True, axis=1)

    # In[22]:

    data = data.fillna(-999)
    data.head()

    # In[23]:

    categorical_columns = [f for f in data.columns]


    data['b14/a1_a3_a4_a19_b1_b12'] = data['B14'] / (
            data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12'])
    data['b14/a1_a3_a4_a19'] = data['B14'] / (data['A1'] + data['A3'] + data['A4'] + data['A19'])
    data['b14/a1_a3_a4'] = data['B14'] / (data['A1'] + data['A3'] + data['A4'])
    data['b14/a1_a3'] = data['B14'] / (data['A1'] + data['A3'])

    data['a1+a3+a4'] = data['A1'] + data['A3'] + data['A4']
    data['a21/a22'] = data['A21'] / data['A22']
    data['a21+a22'] = data['A21'] + data['A22']

    data['b14/a4'] = data['B14'] / data['A4']
    data['b14/a19'] = data['B14'] / data['A19']

    # temperature and pressure
    data['a17/a18'] = data['A17'] / data['A18']
    data['a12/a13'] = data['A12'] / data['A13']
    data['a18/a13'] = data['A18'] / data['A13']
    data['a17-a12'] = data['A17'] - data['A12']

  # 删除某一取值占比超90%的列
    del data['A1']
    del data['A3']
    del data['A4']
    del data['A13']
    del data['A18']
    categorical_columns.remove('A1')
    categorical_columns.remove('A3')
    categorical_columns.remove('A4')
    categorical_columns.remove('A13')
    categorical_columns.remove('A18')

    # In[26]:

    good_cols = [f for f in categorical_columns if f not in (time_period_cols + time_point_list)]


    err_columns = [f for f in good_cols if f not in (time_period_cols + ['样本id', '收率'])]
    best_conditions = data.mean()
    # best_conditions = data.mode().loc[0]
    for col in err_columns:
        data[col + '_err'] = best_conditions[col] - data[col]
        data[col + '_err_abs'] = data[col + '_err'].abs()


    best_conditions = data[data['收率'] >= 0.997].mode().loc[0]
    # err_columns = [f for f in good_cols if f not in (time_period_cols+['样本id','收率'])]
    for col in err_columns:
        data[col + '_err_best'] = best_conditions[col] - data[col]
        data[col + '_err_abs_best'] = data[col + '_err_best'].abs()

    groupList = ['B14', 'A5', 'A19', 'B1', 'B6']
    # groupList = ['B14','A5','A19','B1']
    # err_columns = [f for f in good_cols if f not in (time_period_cols+['样本id','收率'])]
    for group_col in groupList:
        best_conditions = data.groupby(group_col).transform('mean')
        for col in err_columns:
            if col != group_col:
                data[col + '_err' + group_col] = best_conditions[col] - data[col]
                data[col + '_err_abs' + group_col] = data[col + '_err' + group_col].abs()

    # In[30]:

    numerical_columns = [f for f in data.columns if f not in categorical_columns]

    # In[31]:

    data.drop(['收率'], inplace=True, axis=1)
    categorical_columns.remove('收率')

    # In[32]:

    # label encoder
    for f in categorical_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]
    print(train.shape)
    print(test.shape)

    features = numerical_columns + categorical_columns

    # In[35]:

    X_train = train[features].values
    X_test = test[features].values

    y_train = target.values

    # ## train

    train['intTarget'] = pd.cut(y_train, 5, labels=False)

    # In[39]:

    # lgb # faster testing
    param = {'num_leaves': 63,
             #          'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)

    oof_lgb = np.zeros(len(train))  # train data predictions
    predictions_lgb = np.zeros(len(test))  # test data predictions

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, train['intTarget'])):
        print("fold NO.{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                        early_stopping_rounds=100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target) / 2))

    sorted(zip(clf.feature_importance(), features), reverse=True)

    # In[41]:

    # xgb# better
    xgb_params = {'eta': 0.005, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': -1}

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, train['intTarget'])):
        print("fold NO.{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=100,
                        verbose_eval=200, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target) / 2))

    # In[42]:

    sub_df = pd.DataFrame()
    sub_df[0] = test_ids
    sub_df[1] = (predictions_lgb + predictions_xgb) / 2

    if (testFile=='FuSai.csv'):
        sub_df[1] = sub_df[1].apply(lambda x: round(x, 3))
        sub_df.to_csv("./submit_FuSai.csv", index=False, header=None)
    else:
        sub_df.to_csv("./submit_optimize.csv", index=False, header=None)
        print(predictions_lgb,predictions_xgb)

main(testFile='optimize.csv')  # for  optimize.csv
main(testFile='FuSai.csv')  # for FuSai.csv