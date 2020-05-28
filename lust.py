import math
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from dateutil import parser
import numpy as np
import time
import xgboost as xgb
from sklearn.metrics import f1_score
########################################################################################################################
def get_minutes_from_mid_night(st): return parser.parse(st).time().hour * 60 + parser.parse(st).time().minute
def get_df(name): return pd.read_csv(name)
def get_shuffled_df(df): return df.sample(frac=1).reset_index(drop=True)
def drop_empty_rows(df):  df.dropna(how='any',inplace = True)
def process_label_if_available(df):
    if 'label' in df.columns:  df[['label']] = df[['label']].applymap(lambda x: 1 if (x == 'correct' or x ==1 or x =='1') else 0)
def add_distance(df): df['distance'] = ((df['pick_lat']- df['drop_lat'])**2 + (df['pick_lon']- df['drop_lon'])**2)**0.5
def add_log_duration(df): df['log_duration'] = df['duration'].map(lambda x : math.log(max(1,x)))
def add_log_fare(df): df['log_fare'] = df['fare'].map(lambda x : math.log(max(1,x)))
def add_avg_lat(df): df['avg_lat'] = (df['pick_lat']+df['drop_lat'])/2
def add_avg_lon(df): df['avg_lon'] = (df['pick_lon']+df['drop_lon'])/2
def add_log_distance(df): df['log_distance'] = df['distance'].map(lambda x : math.log(max(0.0000001,x)))
def add_meter_reading_fare_capped(df,cap = 300): df['meter_waiting_fare_capped'] = df['meter_waiting_fare'].map(lambda x : min(cap,x))
def add_distance_multiplied(df): df['distance_multiplied'] = df['distance']*1000
def add_time(df): df['time'] = df['pickup_time'].map(lambda x : get_minutes_from_mid_night(x))
def load_train(): return get_processed_df("train.csv")
def load_ops(): return get_processed_df("ops.csv")
def load_original_train(): return get_processed_df("original_train.csv")
def load_original_test(): return get_processed_df("original_test.csv",False)
def load_test(): return get_processed_df("test.csv")

def get_processed_df(name,shuffle=True):
    df = get_df("./data/"+name)
    if shuffle:df = get_shuffled_df(df)
    drop_empty_rows(df)
    process_label_if_available(df)
    add_distance(df)
    add_log_fare(df)
    add_log_duration(df)
    add_log_distance(df)
    add_avg_lat(df)
    add_avg_lon(df)
    add_meter_reading_fare_capped(df)
    add_distance_multiplied(df)
    add_time(df)
    df = df[[col for col in df.columns if col != 'fare'] + ['fare']] # change fare column position to end
    if 'label' in df.columns:df = df[[col for col in df.columns if col != 'label'] + ['label']] # change label column position to end
    df.reset_index(drop=True, inplace=True)
    return df


def select_input_columns_regress(df): return df[input_columns_regress]
def select_output_columns_as_row_regress(df): return df[output_columns_regress[0]].ravel()

def select_input_columns_classify(df): return df[input_columns_classify]
def select_output_columns_as_row_classify(df): return df[output_columns_classify[0]].ravel()

def split_n_save(tr = .6, ts =.2,name='original_train.csv'):
    df = pd.read_csv('./data/'+name)
    basernd = np.random.rand(len(df))
    train_mask = basernd <= tr
    test_mask = basernd > (1- ts)
    ops_mask = (basernd >tr) & (basernd < (1- ts))
    train = df[train_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)
    ops = df[ops_mask].reset_index(drop=True)
    train.to_csv('./data/train.csv')
    test.to_csv('./data/test.csv')
    ops.to_csv('./data/ops.csv')

def split_df(df,p):
    basernd = np.random.rand(len(df))
    train_mask = basernd < p
    train = df[~train_mask].reset_index(drop=True)
    test = df[train_mask].reset_index(drop=True)
    return train,test

def pd_vstack(dfs):
    big_list = None
    for i in dfs:
        if big_list is None: big_list = i
        else:big_list = big_list.append(i, ignore_index=True)
    big_list.reset_index(drop=True, inplace=True)
    return big_list

def remove_label_false(df): return df[df['label'] > .5]
def remove_label_true(df): return df[df['label'] < .5]

########################################################################################################################
input_columns_regress = [

    #'fare',
    'meter_waiting_fare',
    #'meter_waiting',
    #'meter_waiting_till_pickup',
    'distance_multiplied',
    'duration',
    'pick_lat',
    'pick_lon',
    # 'drop_lat',
    # 'drop_lon',
    # 'avg_lat',
    # 'avg_lon',
    'additional_fare',
    'time',

#    'log_fare',
#    'log_duration',

#    'predicted_fare',
#    'predicted_fare_error_perc',
#    'predicted_fare_error_diff'
]

output_columns_regress = ['fare']

######################################
input_columns_classify = [

    'fare',
#    'pfare',
    'meter_waiting_fare',
    #'meter_waiting',
    #'meter_waiting_till_pickup',
    'distance_multiplied',
    'duration',
    'pick_lat',
    'pick_lon',
    # 'drop_lat',
    # 'drop_lon',
    # 'avg_lat',
    # 'avg_lon',
    'additional_fare',
    'time',

#    'log_fare',
#    'log_duration',

#    'predicted_fare',
#    'predicted_fare_error_perc',
#    'predicted_fare_error_diff'
]

output_columns_classify = ['label']

########################################################################################################################
# tr_df = load_train()
# ts_df = load_test()


tr, ts = split_df(load_original_train(), .1)
oo = load_original_test()
########################################################################################################################
# def xgb_f1(y, t, threshold=0.5):
#     t = t.get_label()
#     y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
#     return 'f1',f1_score(t,y_bin)
#
#
# mod = xgb.XGBClassifier(n_estimators=10000,learning_rate=.4)
#
#
# eval_set = [(select_input_columns(ts), select_output_columns_as_row(ts))]
#
# mod.fit(select_input_columns(tr), select_output_columns_as_row(tr) ,eval_metric=xgb_f1, eval_set=eval_set, verbose=True)

#g=f1_score(select_output_columns_as_row(ts),mod.predict(select_input_columns(ts)))

########################################################################################################################
def r():
    clas = CatBoostClassifier(iterations=3000, eval_metric='F1')
    reg = CatBoostRegressor(iterations=29, eval_metric='MAE')
    reg.fit(

                    select_input_columns_regress(pd_vstack([tr, ts,oo])),
                    select_output_columns_as_row_regress(pd_vstack([tr, ts,oo])),

            eval_set=[((select_input_columns_regress(remove_label_true(pd_vstack([tr, ts])))),
                      select_output_columns_as_row_regress(remove_label_true(pd_vstack([tr, ts])))),
                      ((select_input_columns_regress(remove_label_false(pd_vstack([tr, ts])))),
                       select_output_columns_as_row_regress(remove_label_false(pd_vstack([tr, ts])))),
                      ],
            use_best_model=True, verbose=True)
    ts['pfare'] = pd.Series(reg.predict(select_input_columns_regress(ts)))
    tr['pfare'] = pd.Series(reg.predict(select_input_columns_regress(tr)))
    oo['pfare'] = pd.Series(reg.predict(select_input_columns_regress(oo)))
    clas.fit(select_input_columns_classify(tr), select_output_columns_as_row_classify(tr), eval_set=(select_input_columns_classify(ts), select_output_columns_as_row_classify(ts)), use_best_model=True, verbose=True)


    ts['plabel'] = pd.Series(clas.predict(select_input_columns_classify(ts)))

    oo['prediction']=pd.Series(clas.predict(select_input_columns_classify(oo)))
    fname = str(clas.best_score_['validation']['F1'])+"_"+str(int(time.time() * 1000))+".csv"
    oo[['tripid','prediction']].to_csv("./results/"+fname,index=False)

    print(fname)