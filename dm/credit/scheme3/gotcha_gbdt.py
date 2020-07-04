import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
from sklearn.linear_model import HuberRegressor
import sklearn.ensemble as tree_model
from tqdm import tqdm
import datetime
pd.set_option('display.max_column',100)
warnings.filterwarnings('ignore')
from utils import make_dir, score, timer, kf_lgbm, kf_xgbm, kf_ctbm, kf_sklearn

import sklearn


def make_features(df):
    app_feature = [
        '当月网购类应用使用次数',
        '当月物流快递类应用使用次数',
        '当月金融理财类应用使用总次数',
        '当月视频播放类应用使用次数',
        '当月飞机类应用使用次数',
        '当月火车类应用使用次数',
        '当月旅游资讯类应用使用次数',
    ]

    for f in app_feature:
        df['round_log1p' + f] = np.round(np.log1p(df[f])).astype(int)

    df['前五个月消费总费用'] = 6 * df['用户近6个月平均消费值（元）'] - df['用户账单当月总费用（元）']
    df['前五个月消费平均费用'] = df['前五个月消费总费用'] / 5
    df['当月费用/前五个月消费平均费用'] = (df['用户账单当月总费用（元）']) \
                            / (1 + df['前五个月消费平均费用'])
    df['当月费用-前五个月消费平均费用'] = df['用户账单当月总费用（元）'] - df['前五个月消费平均费用']

    def make_count_feature(df, col, fea_name):
        df['idx'] = range(len(df))
        tmp = df.groupby(col)['用户编码'].agg([
            (fea_name, 'count')]).reset_index()
        df = df.merge(tmp)
        df = df.sort_values('idx').drop('idx', axis=1).reset_index(drop=True)
        return df

    df = make_count_feature(df, '缴费用户最近一次缴费金额（元）', 'count_缴费')
    df = make_count_feature(df, '用户账单当月总费用（元）', 'count_当月费用')
    df = make_count_feature(df, '前五个月消费总费用', 'count_总费用')
    df = make_count_feature(df, '当月费用-前五个月消费平均费用', 'count_费用差')
    df = make_count_feature(df, '用户近6个月平均消费值（元）', 'count_平均费用')
    df = make_count_feature(df, ['用户账单当月总费用（元）', '用户近6个月平均消费值（元）'],
                            'count_当月费用_平均费用')

    arr = df['缴费用户最近一次缴费金额（元）']
    df['是否998折'] = ((arr / 0.998) % 1 == 0) & (arr != 0)

    df['年龄_0_as_nan'] = np.where(df['用户年龄'] == 0, [np.nan] * len(df), df['用户年龄'])

    return df


def load_df_and_make_features():
    train_df = pd.read_csv('../input/train_dataset.csv')
    test_df = pd.read_csv('../input/test_dataset.csv')
    train_df['train'] = 1
    test_df['train'] = 0
    df = pd.concat([train_df, test_df])
    df = make_features(df)
    return df

feature_name1 = \
['用户年龄',
 '用户网龄（月）',
 '用户实名制是否通过核实',
 '是否大学生客户',
 '是否4G不健康客户',
 '用户最近一次缴费距今时长（月）',
 '缴费用户最近一次缴费金额（元）',
 '用户近6个月平均消费值（元）',
 '用户账单当月总费用（元）',
 '用户当月账户余额（元）',
 '用户话费敏感度',
 '当月费用-前五个月消费平均费用',
 '前五个月消费总费用',
 'count_缴费',
 'count_当月费用',
 'count_费用差',
 'count_平均费用',
 'count_当月费用_平均费用',
 '是否998折',
 '当月通话交往圈人数',
 '近三个月月均商场出现次数',
 '当月网购类应用使用次数',
 '当月物流快递类应用使用次数',
 '当月金融理财类应用使用总次数',
 '当月视频播放类应用使用次数',
 '当月飞机类应用使用次数',
 '当月火车类应用使用次数',
 '当月旅游资讯类应用使用次数',
 '当月是否逛过福州仓山万达',
 '当月是否到过福州山姆会员店',
 '当月是否看电影',
 '当月是否景点游览',
 '当月是否体育场馆消费',
 '是否经常逛商场的人',
 '是否黑名单客户',
 '缴费用户当前是否欠费缴费']

df = load_df_and_make_features()
train_df = df[df['train']==1]
test_df = df[df['train']!=1]

output_dir = './stacking_files/'

x, y = train_df[feature_name1], train_df['信用分'].values
x_test = test_df[feature_name1]

model = kf_sklearn(x=x,y=y,x_test=x_test,output_dir=output_dir,name="gotcha_gbdt1",
                   model_class=tree_model.GradientBoostingRegressor,
                   n_estimators=250,subsample=0.8,min_samples_leaf=10,
                   max_depth=7,loss='huber',random_state=2019)