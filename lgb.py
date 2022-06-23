#!/usr/bin/env python
# coding: utf-8

# In[133]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')


# In[134]:


path = 'C:/Users/gusals/Desktop/현민/운동동작분류/'

train=pd.read_csv(path + 'open/train_features.csv')
train_label=pd.read_csv(path + 'open/train_labels.csv')
test=pd.read_csv(path + 'open/test_features.csv')
submission=pd.read_csv(path + 'open/sample_submission.csv')


# In[135]:


train.shape, train_label.shape, test.shape, submission.shape


# In[136]:


train.head()


# In[137]:


train[['id', 'time']].groupby('id').count()


# In[138]:


def plot_acc_id(df, id_num):
    return df.query('id == @id_num')[['time', 'acc_x', 'acc_y', 'acc_z']].plot(x = 'time')

def plot_gy_id(df, id_num):
    return df.query('id == @id_num')[['time', 'gy_x', 'gy_y', 'gy_z']].plot(x = 'time')


# In[139]:


features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
X_train = train[features].groupby('id').agg(['max', 'min', 'mean'])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean'])


# In[140]:


X_train.shape, X_test.shape


# In[141]:


X_train.head()


# In[142]:


y_train = train_label['label']


# In[143]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.2, random_state=0)


# In[144]:


bayes_params = {
    'num_leaves': (24, 45),
    'colsample_bytree':(0.5, 1), 
    'subsample': (0.5, 1),
    'max_depth': (4, 12),
    'reg_alpha': (0, 0.5),
    'reg_lambda': (0, 0.5), 
    'min_split_gain': (0.001, 0.1),
    'min_child_weight':(5, 50)
}


# In[145]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def lgb_roc_eval(num_leaves, colsample_bytree, subsample, max_depth, reg_alpha, reg_lambda, min_split_gain, min_child_weight):
    
    params = {
        "n_estimator":200,
        "learning_rate":0.02,
        'num_leaves': int(round(num_leaves)),
        'colsample_bytree': colsample_bytree, 
        'subsample': subsample,
        'max_depth': int(round(max_depth)),
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda, 
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'verbosity': -1,
        'objective': 'multiclass'
    }
    print("params:", params)
    lgb_model = LGBMClassifier(**params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, 
                  eval_metric="multi_logloss", verbose=100 )
    best_iter = lgb_model.best_iteration_
    print('best_iter:', best_iter)
    valid_proba = lgb_model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    roc_preds = roc_auc_score(y_test, valid_proba)
    print('roc_auc:', roc_preds)
    return roc_preds


# In[146]:



from bayes_opt import BayesianOptimization

BO_lgb = BayesianOptimization(lgb_roc_eval, bayes_params, random_state=0)


# In[147]:


BO_lgb.maximize(init_points=5, n_iter=10)


# In[ ]:




