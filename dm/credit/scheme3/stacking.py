import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.isotonic import IsotonicRegression
import sklearn.linear_model as linear_model
import sklearn.ensemble as tree_model
import sklearn.svm as svm
from utils import make_dir, score, timer, kf_lgbm
pd.set_option('display.max_column',100)

def load(meta_dir, filename):
    return np.load(os.path.join(meta_dir, filename))

def load_val_test(name, input_dir='./stacking_files/'):
    val = load(input_dir, f'val.{name}.npy')
    test = load(input_dir, f'test.{name}.npy')
    return val, test

liuxin_stack_files = [
                     'test.justai_ctb.npy',
                     'test.justai_lgb.npy',
                     'test.luoling_ctb.npy',
                     'test.luoling_xgb.npy',
                     'val.justai_ctb.npy',
                     'val.justai_lgb.npy',
                     'val.luoling_ctb.npy',
                     'val.luoling_xgb.npy']
for f in liuxin_stack_files:
    npf = np.load(os.path.join('./liuxin/stack/',f))
    np.save(os.path.join('./stacking_files/',f),npf)

a = np.load('./stacking_files/test.neil_lgb.npy')
b = np.load('./stacking_files/val.neil_lgb.npy')
np.save('./stacking_files/test.neil_lgb_rounded.npy',np.round(a))
np.save('./stacking_files/val.neil_lgb_rounded.npy',np.round(b))

a = np.load('./stacking_files/test.neil_xgb.npy')
b = np.load('./stacking_files/val.neil_xgb.npy')
np.save('./stacking_files/test.neil_xgb_rounded.npy',np.round(a))
np.save('./stacking_files/val.neil_xgb_rounded.npy',np.round(b))

name_list = [
    'neil_lgb_rounded',
    'neil_ctb',
    'neil_rf',
    'gotcha_lgb1',
    'gotcha_lgb2',
    'gotcha_lgb3',
    'gotcha_lgb4',
    'gotcha_lgb5',
    'justai_lgb',
    'justai_ctb',
    'gotcha_lgb6',
    'gotcha_ctb1',
    'luoling_xgb',
    'luoling_ctb',
    'neil_xgb_rounded',
    'gotcha_gbdt1',
    'neil_gbm',
]
val_list = []
test_list = []
for name in name_list:
    val, test = load_val_test(name)
    val_list.append(val)
    test_list.append(test)

X = np.stack(val_list, axis=1)
X_test = np.stack(test_list, axis=1)

train_df = pd.read_csv('../input/train_dataset.csv')
test_df = pd.read_csv('../input/test_dataset.csv')
y = train_df['信用分'].values

n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=8888)
idx = y.argsort()
y_lab = np.repeat(list(range(2500)),20)
y_lab = np.asarray(sorted(list(zip(idx,y_lab))))[:,-1].astype(np.int32)

test_pred = np.zeros(len(X_test))
val_pred = np.zeros(len(X))
scores = []
for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X,y_lab)):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    model = linear_model.HuberRegressor(epsilon=1.01, alpha=1e-5)
    model.fit(X_train,y_train)
    val_pred_fold = model.predict(X_valid)
    val_pred[valid_idx] = val_pred_fold
    s = score(np.round(val_pred_fold),y_valid)
    scores.append(s)
    test_pred += model.predict(X_test)/10
    print(model.coef_)
    print(np.round(s,8))


def postprocess(pred):
    print('=' * 10, 'postprocess', '=' * 10)
    new_pred = pred.copy()

    mask = (pred < 619)
    new_pred = np.where(mask, np.round(pred + 0.1), new_pred)
    num_change_value = (np.round(new_pred) != np.round(pred))[mask].sum()
    print('%d/%d values are changed' % (num_change_value, sum(mask)))

    mask = (pred > 619)
    new_pred = np.where(mask, np.round(pred + 0.02), new_pred)
    num_change_value = (np.round(new_pred) != np.round(pred))[mask].sum()
    print('%d/%d values are changed' % (num_change_value, sum(mask)))

    new_pred = np.round(new_pred).astype(int)

    return new_pred


print('未round:',score(val_pred, y))
print('round:',score(np.round(val_pred), y))
print('后处理（小于619,+0.1, 大于619, +0.02）:',score(postprocess(val_pred), y))

sub = pd.read_csv('../input/submit_example.csv')
sub[' score'] = np.round(postprocess(test_pred)).astype(int)

sub.to_csv('cv0.644190_pb64107.csv',index=False)

sub.head()