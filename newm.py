from subprocess import call, Popen
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import time
from dateutil import parser
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
def load_keras():
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers.core import Dense
    import keras
    from keras import backend as K

# Pre Processing methods -----------------------------------------------------------------------------------------------
def xc(df):
    fname = str(int(time.time() * 1000))
    df.to_csv("./tmp/"+fname+".csv")
    Popen("C:\Program Files (x86)\Microsoft Office/root\Office16/excel.exe ./tmp/"+fname+".csv")
def get_df(name):
    return pd.read_csv(name)
def get_df_na(name):
    return get_df(name).dropna(how='any')
def process_label_if_available(df):
    df[['label']] = df[['label']].applymap(lambda x: x == 'correct')
def add_distance(df):
    df['distance'] = ((df['pick_lat']- df['drop_lat'])**2 + (df['pick_lon']- df['drop_lon'])**2)**0.5
def add_log_duration(df):
    df['log_duration'] = df['duration'].map(lambda x : math.log(max(1,x)))
def add_log_fare(df):
    df['log_fare'] = df['fare'].map(lambda x : math.log(max(1,x)))
def add_log_distance(df):
    df['log_distance'] = df['distance'].map(lambda x : math.log(max(0.0000001,x)))
def Y(df): return df.shape[0]
def remove_incorrect(df):
    df.drop(df[df.label < 0.5].index, inplace=True)
def remove_correct(df):
    df.drop(df[df.label > 0.5].index, inplace=True)

def add_time(df):
    df

# Program --------------------------------------------------------------------------------------------------------------
df = get_df_na("./data/train.csv")
process_label_if_available(df)
add_distance(df)
add_log_fare(df)
add_log_duration(df)
add_log_distance(df)

