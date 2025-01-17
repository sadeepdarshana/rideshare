import math
import time
from subprocess import Popen

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dateutil import parser
import joblib
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Constants ------------------------------------------------------------------------------------------------------------
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


input_columns = None
output_columns = None
no_norm = True


input_columns_regress = [

    # 'log_fare',
    'meter_waiting_fare',
    'meter_waiting',
    'meter_waiting_till_pickup',
    'distance_multiplied',
    'duration',
    'pick_lat',
    'pick_lon',
    'drop_lat',
    'drop_lon',
    'additional_fare',
    #  'log_duration',
    #'time',
    #  'predicted_fare_error_perc',
    #   'predicted_fare_error_diff'
]

output_columns_regress = [
                 'fare'
]

input_columns_classify = [
                 'fare',
              #   'predicted_fare',
               #  'log_fare',
                 'meter_waiting_fare',
  #               'meter_waiting',
              #   'meter_waiting_till_pickup',
                 'distance_multiplied',
                 'duration',
  #               'pick_lat',
  #               'pick_lon',
                 # 'drop_lat',
                 # 'drop_lon',
                 # 'avg_lat',
                 # 'avg_lon',
                 'additional_fare',
               #  'log_duration',
              #   'time',
                # 'predicted_fare_error_perc',
               #  'predicted_fare_error_diff'
]

output_columns_classify = [
                 'label'
]

# Build model ----------------------------------------------------------------------------------------------------------



# Train ----------- ----------------------------------------------------------------------------------------------------
def get_train_model_inputs_output(df_with_features_n_label):
    df_in = select_input_columns(df_with_features_n_label)
    df_out = select_output_columns(df_with_features_n_label)

    normalizer = build_normalizer(df_in, 'min_max')
    df_in_normalized = transform_with(df_in, normalizer)

    sci_model_in = format_to_scikit_input(df_in_normalized)
    sci_model_out = format_to_scikit_output(df_out)

    return sci_model_in,sci_model_out,normalizer

def train_n_get_normalizer(df_with_features_n_label, model):
    sci_model_in, sci_model_out, normalizer = get_train_model_inputs_output(get_shuffled_df(df_with_features_n_label))


    basernd = np.random.rand(len(sci_model_in))
    train_mask = basernd < .9

    if input_columns_regress == input_columns:
        model.fit(sci_model_in[train_mask], sci_model_out[train_mask], eval_set=(sci_model_in[~train_mask], sci_model_out[~train_mask]),use_best_model=True,verbose=True)
    else:
        model.fit(sci_model_in[train_mask], sci_model_out[train_mask],eval_set=(sci_model_in[~train_mask], sci_model_out[~train_mask]), use_best_model=True, verbose=True)

    return normalizer


# Predict --------------------------------------------------------------------------------------------------------------
def get_predict_model_inputs_output(df_with_features,normalizer):
    df_in = select_input_columns(df_with_features)
    df_in_normalized = transform_with(df_in, normalizer)
    sci_model_in = format_to_scikit_input(df_in_normalized)
    return sci_model_in


def predict_n_build_column(model, df_with_features, normalizer, output_column_name ='predicted'):
    sci_model_in = get_predict_model_inputs_output(df_with_features,normalizer)
    pred = model.predict(sci_model_in)
    df_with_features[output_column_name] = pd.Series(pred.reshape(len(pred)))


def add_predicted_fare_error_perc(df):
    df['predicted_fare_error_perc'] = (df['predicted_fare'] - df['fare'])/ df['fare'].clip(lower = 20)

def add_predicted_fare_error_diff(df):
    df['predicted_fare_error_diff'] = (df['predicted_fare'] - df['fare'])


def predict_label_based_on_predicted_fare(df,cut_off_lb,cut_off_ub):
    df['predicted_label'] = (cut_off_lb < df['predicted_fare_error_perc']) & (df['predicted_fare_error_perc'] < cut_off_ub)

# Utils ----------------------------------------------------------------------------------------------------------------

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

def get_minutes_from_mid_night(st): return parser.parse(st).time().hour * 60 + parser.parse(st).time().minute

def select_input_columns(df): return df[input_columns]

def select_output_columns(df): return df[output_columns]

def format_to_scikit_input(df_normalized): return df_normalized.values

def format_to_scikit_output(df): return df.values.reshape(len(df),)

def pd_vstack(dfs):
    big_list = None
    for i in dfs:
        if big_list is None: big_list = i
        else:big_list = big_list.append(i, ignore_index=True)
    big_list.reset_index(drop=True, inplace=True)
    return big_list

def xc(df):
    fname = str(int(time.time() * 1000))
    df.to_csv("./tmp/"+fname+".csv")
    Popen("C:\Program Files (x86)\Microsoft Office/root\Office16/excel.exe ./tmp/"+fname+".csv")

# Load data ------------------------------------------------------------------------------------------------------------
def get_df(name): return pd.read_csv(name)

def get_processed_df(name):
    unshuffled_df = get_df("./data/"+name)
    df = get_shuffled_df(unshuffled_df)
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

def load_train(): return get_processed_df("train.csv")

def load_ops(): return get_processed_df("ops.csv")

def load_original_train(): return get_processed_df("original_train.csv")

def load_test(): return get_processed_df("test.csv")

# Pre process data -----------------------------------------------------------------------------------------------------
def get_shuffled_df(df): return df.sample(frac=1).reset_index(drop=True)

def drop_empty_rows(df):  df.dropna(how='any',inplace = True)

def process_label_if_available(df):
    if 'label' in df.columns:
        df[['label']] = df[['label']].applymap(lambda x: 1 if (x == 'correct' or x ==1 or x =='1') else 0)

def add_distance(df): df['distance'] = ((df['pick_lat']- df['drop_lat'])**2 + (df['pick_lon']- df['drop_lon'])**2)**0.5

def add_log_duration(df): df['log_duration'] = df['duration'].map(lambda x : math.log(max(1,x)))

def add_log_fare(df): df['log_fare'] = df['fare'].map(lambda x : math.log(max(1,x)))

def add_avg_lat(df): df['avg_lat'] = (df['pick_lat']+df['drop_lat'])/2

def add_avg_lon(df): df['avg_lon'] = (df['pick_lon']+df['drop_lon'])/2

def add_log_distance(df): df['log_distance'] = df['distance'].map(lambda x : math.log(max(0.0000001,x)))

def add_meter_reading_fare_capped(df):
    cap = 300
    df['meter_waiting_fare_capped'] = df['meter_waiting_fare'].map(lambda x : min(cap,x))

def add_distance_multiplied(df): df['distance_multiplied'] = df['distance']*1000

def add_time(df): df['time'] = df['pickup_time'].map(lambda x : get_minutes_from_mid_night(x))

# Scaling / Standardising ----------------------------------------------------------------------------------------------
def build_normalizer(df, name="", normalizer_type ='min_max'):
    normalizer = None
    if normalizer_type == 'min_max': normalizer = MinMaxScaler()
    if normalizer_type == 'std': normalizer = StandardScaler()
    normalizer.fit(df)
    if name:joblib.dump(normalizer, './normalizer'+name)
    return normalizer

def transform_with(df, norm_model):
    if isinstance(norm_model, str):normalizer = joblib.load('./normalizer' + norm_model)
    else: normalizer = norm_model
    normed =  pd.DataFrame(normalizer.transform(df), columns=df.columns)
    if no_norm:return df
    ks = 'predicted_fare_error_perc','predicted_fare_error_diff'

    for k in ks:
        if k in df.columns:
            col = df[k]
            maxv = max(col.max(),-1*col.min(),1)
            newcol = col/maxv
            normed[k] = newcol

    return normed



# ----------------------------------------------------------------------------------------------------------------------

def calc_matrices(df_with_class):
    raw_matrices = df_with_class['class'].value_counts()
    tp, tn, fp, fn = (raw_matrices[1] if 1 in raw_matrices else 0,
                      raw_matrices[2] if 2 in raw_matrices else 0,
                      raw_matrices[3] if 3 in raw_matrices else 0,
                      raw_matrices[4] if 4 in raw_matrices else 0)
    matrices = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp)                 if tp else 0,
        'recall': tp / (tp + fp)                    if tp else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if tp+tn else 0,
        'f1':2*tp/(2*tp+fn+fp)                      if tp else 0
    }
    return pd.DataFrame([matrices])

def attach_classes(df):
    predicted = df['predicted_label']
    actual = df['label']
    tp = (predicted == actual) & (predicted == True)
    tn = (predicted == actual) & (predicted == False)
    fp = (predicted != actual) & (predicted == True)
    fn = (predicted != actual) & (predicted == False)
    df['class'] = tp * 1 + tn * 2 + fp * 3 + fn * 4

def set_mode_classify():
    global input_columns
    global output_columns
    input_columns = input_columns_classify
    output_columns = output_columns_classify

def set_mode_regress():
    global input_columns
    global output_columns
    input_columns = input_columns_regress
    output_columns = output_columns_regress

def produce(test_df):
    fname = str(int(time.time() * 1000))+'.csv'
    test_df['prediction'] = test_df['predicted_label']
    r = test_df[['tripid','prediction']]
    r.sort_values('tripid',inplace=True)
    r.to_csv('./results/'+fname,index=False)
#-----------------------------------------------------------------------------------------------------------------------

def all(regressor_model = None, classifier_model = None,m1=-1,M1=300000000,m2=-1,M2=300000000,no_test=False):
    train_df = load_train()
    test_df = load_test()
    ops_df = load_ops()

    m1-=0.00001
    m2-=0.00001
    M1+=0.00001
    M2+=0.00001
    train_df = train_df[train_df['fare'] > m1]
    test_df = test_df[test_df['fare'] > m1]
 #   ops_df = ops_df[ops_df['fare'] > m1]

    train_df = train_df[train_df['fare'] < M1]
    test_df = test_df[test_df['fare'] < M1]
 #   ops_df = ops_df[ops_df['fare'] < M1]

    train_df.reset_index(drop=True,inplace=True)
    test_df.reset_index(drop=True,inplace=True)
    ops_df.reset_index(drop=True,inplace=True)

    ops_df_stripped_false = ops_df[ops_df['label'] > .5]
    set_mode_regress()#set mode
    #regressor_model = RandomForestRegressor(n_estimators=20, random_state=0)
    normalizer = train_n_get_normalizer(pd_vstack([train_df,test_df]), regressor_model) # train regress on train
    predict_n_build_column(regressor_model, train_df, normalizer, output_column_name='predicted_fare') # predict regress on train
    predict_n_build_column(regressor_model, test_df, normalizer, output_column_name='predicted_fare') # predict regress on test
    add_predicted_fare_error_perc(train_df)
    add_predicted_fare_error_perc(test_df)
    add_predicted_fare_error_diff(train_df)
    add_predicted_fare_error_diff(test_df)
    set_mode_classify()
    #classifier_model = RandomForestClassifier(n_estimators=100, random_state=0)
    normalizer = train_n_get_normalizer(train_df, classifier_model) #train classifier on train
    predict_n_build_column(classifier_model, test_df, normalizer, output_column_name='predicted_label')

    if no_test:
        produce(test_df)
        return test_df
    attach_classes(test_df)


    test_df = test_df[test_df['fare'] > m2]
    test_df = test_df[test_df['fare'] < M2]
    test_df.reset_index(drop=True,inplace=True)

    m=calc_matrices(test_df)
    return test_df,m

def splitrun():
    #split_n_save(.99, .01)
    g=all(CatBoostRegressor(iterations=30), CatBoostClassifier(iterations=10000,eval_metric='F1'),m1=-1,M1=26444444404,m2=-1,M2=233333333360)
    print(g[1])

splitrun()

#all(RandomForestRegressor(n_estimators=40, random_state=42), RandomForestClassifier(n_estimators=130))
#all(RandomForestRegressor(n_estimators=100), RandomForestClassifier(n_estimators=100))
if 0:
    regressor_model = RandomForestRegressor(n_estimators=40, random_state=42)
    train_df = load_train()
    test_df = load_test()
    ops_df = load_ops()
    ops_df_stripped_false = ops_df[ops_df['label'] > .5]
    set_mode_regress()#set mode
    #regressor_model = RandomForestRegressor(n_estimators=20, random_state=0)
    normalizer = train_n_get_normalizer(ops_df_stripped_false, regressor_model) # train regress on train

    predict_n_build_column(regressor_model, train_df, normalizer, output_column_name='predicted_fare') # predict regress on train
    predict_n_build_column(regressor_model, test_df, normalizer, output_column_name='predicted_fare') # predict regress on test
    predict_n_build_column(regressor_model, ops_df, normalizer, output_column_name='predicted_fare') # predict regress on test
    add_predicted_fare_error_perc(train_df)
    add_predicted_fare_error_perc(test_df)
    add_predicted_fare_error_perc(ops_df)
    add_predicted_fare_error_diff(train_df)
    add_predicted_fare_error_diff(test_df)
    add_predicted_fare_error_diff(ops_df)


#
# test_df['predicted_label'] = test_df['predicted_label'] *1
# test_df['predicted_fare'] = test_df['predicted_fare'] *1
# test_df['predicted_fare_error_perc'] = test_df['predicted_fare_error_perc'] *1

#
#
# model = get_a_new_sci_model()
#
# normalizer = train_cl_n_get_normalizer(train_df, model)
#
# predict_cl_n_build_column(model, test_df, normalizer)
# attach_classes(test_df)
# m = calc_matrices(test_df)
# print(m)