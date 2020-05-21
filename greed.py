from subprocess import call, Popen, CREATE_NEW_CONSOLE
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
import sys
import math
import time
from dateutil import parser
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


#-----------------------------------------------------------------------------------------------------------------------
def xc(df):
    fname = str(int(time.time() * 1000))
    df.to_csv("./tmp/"+fname+".csv")
    Popen("C:\Program Files (x86)\Microsoft Office/root\Office16/excel.exe ./tmp/"+fname+".csv")
def mini(st):
    d = parser.parse(st)
    return d.time().hour * 60 + d.time().minute
def get_df(name):
    return pd.read_csv(name)
def get_df_na(name): return get_df(name).dropna(how='any')
def process_label_if_available(df):
    df[['label']] = df[['label']].applymap(lambda x: 1 if (x == 'correct' or x ==1 or x =='1') else 0)
def add_distance(df):
    df['distance'] = ((df['pick_lat']- df['drop_lat'])**2 + (df['pick_lon']- df['drop_lon'])**2)**0.5
def add_log_duration(df):
    df['log_duration'] = df['duration'].map(lambda x : math.log(max(1,x)))
def add_log_fare(df):
    df['log_fare'] = df['fare'].map(lambda x : math.log(max(1,x)))
def add_log_distance(df):
    df['log_distance'] = df['distance'].map(lambda x : math.log(max(0.0000001,x)))
def add_meter_reading_fare_capped(df):
    cap = 300
    df['meter_waiting_fare_capped'] = df['meter_waiting_fare'].map(lambda x : min(cap,x))
def add_distance_multiplied(df):
    df['distance_multiplied'] = df['distance']*1000
def Y(df): return df.shape[0]
def remove_incorrect(df):
    df.drop(df[df.label < 0.5].index, inplace=True)
def remove_correct(df):
    df.drop(df[df.label > 0.5].index, inplace=True)
def add_time(df):
    df['time'] = df['pickup_time'].map(lambda x : mini(x))
def add_color(df):
    df['color'] = df['label'].map(lambda x : 'g' if x else 'r')
def load_train_train():
    return load_train("train_train.csv")
def load_train_test():
    return load_train("train_test.csv")
def load_train(name):
    df = get_df_na("./data/"+name).sample(frac=1).reset_index(drop=True)
    process_label_if_available(df)
    add_distance(df)
    add_log_fare(df)
    add_log_duration(df)
    add_log_distance(df)
    add_meter_reading_fare_capped(df)
    add_distance_multiplied(df)
    add_time(df)
    add_color(df)
    return df

input_columns = ['fare','log_fare','meter_waiting_fare_capped','distance_multiplied','duration','additional_fare','log_duration','time']
output_columns = ['label']
def select_input_columns(df):
    return df[input_columns]
def select_output_columns(df):
    return df[output_columns]
def split_train_test(df,p=0.8):
    msk = np.random.rand(len(df)) < p
    train = df[msk]
    test = df[~msk]
    df_in_tr,df_out_tr = select_input_columns(train),select_output_columns(train)
    df_in_ts,df_out_ts = select_input_columns(test ),select_output_columns(test)
    return df_in_tr,df_out_tr,df_in_ts,df_out_ts
def load_keras_model(name):return load_model("./models/"+name)
def train_save(df,model,epochs,name="", new_optimizer = False):
    df_in, df_out = select_input_columns(df),select_output_columns(df)
    scaler = MinMaxScaler()
    df_in = scaler.fit_transform(df_in)
    if new_optimizer:
        op = Adam()
        model.compile(loss='binary_crossentropy',
                      optimizer=op,
                      metrics=['accuracy'])
    model.fit(df_in, df_out.values,epochs=epochs)
    if name != "" :model.save("./models/"+name)
def build_model():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(len(input_columns),)))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(len(output_columns), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def pd_hstack(*arg):
  return pd.concat(list(arg),axis=1)
def predict(df, model,scaler_fit=None):
    inp = select_input_columns(df)
    scaler = MinMaxScaler()
    if scaler_fit is not None != None:
        scaler.fit(scaler_fit)
    else : scaler.fit(inp)
    inp = scaler.transform(inp)
    predicted_scores = pd.DataFrame.from_records(model.predict(inp))
    return predicted_scores
def get_classes(df_with_output_column, predicted_scores, p=.5):
    out = select_output_columns(df_with_output_column)
    predicted = (predicted_scores > p)
    actual = out > .5
    predicted.rename(columns=(lambda x: 'class'), inplace=True)
    actual.rename(columns=(lambda x: 'class'), inplace=True)
    tp = (predicted == actual) & (predicted == True)
    tn = (predicted == actual) & (predicted == False)
    fp = (predicted != actual) & (predicted == True)
    fn = (predicted != actual) & (predicted == False)
    classes = tp * 1 + tn * 2 + fp * 3 + fn * 4
    return pd_hstack(classes)
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
def get_classes_n_build_matrics_df(actual_labels,predicted_scores,split_value):
    row = pd.DataFrame(calc_matrices(get_classes(actual_labels[['label']], predicted_scores,split_value)))
    row['split_value'] = split_value
    return row
fare_groups_info = [ 75, 125, 175, 275, 375, 475, 575, 775, 1100, 1000000]
def find_group(group_info,n):
    for i in range(len(group_info)):
        if n < group_info[i]: return i
def train_cluster(base_model_name, groups, epochs_per_model, id = None,group_number = -1, new_optimizer = False,no_base_model = False):
    if id is None :id = str(int(time.time() * 1000))
    Path("./models/cluster_" + id).mkdir(parents=True, exist_ok=True)
    for i in range(len(groups)):
        if group_number != -1 and group_number != i:continue
        if no_base_model:
            model = build_model()
        else :
            model = load_keras_model(base_model_name)
        print("training model: "+str(i))
        train_save(groups[i], model, epochs_per_model, "cluster_" + id + "/" + str(i),new_optimizer=new_optimizer)
def add_fare_group(df):
    df['fare_group'] = df['fare'].map(lambda x: find_group(fare_groups_info, x))
def get_fare_groups(df):
    add_fare_group(df)
    return [df.groupby('fare_group').get_group(i).reset_index() for i in range(len(fare_groups_info))]
def predict_n_attach_class_score(df, model_name):
    pred = predict(df, load_keras_model(model_name))
    return pd_hstack(df,get_classes(df, pred),pred)
def get_cluster(id,group_count):
    return [load_keras_model( "cluster_" + id + "/" + str(i)) for i in range(group_count)]
def predict_groups_and_calc_optimal_split(cluster_id, df,scaler_fit = None):
    models = get_cluster(cluster_id, len(fare_groups_info))
    groups = get_fare_groups(df)
    if scaler_fit is not None:scaler_fit_groups = get_fare_groups(scaler_fit)
    splits = []
    for i in range(len(fare_groups_info)):
        model = models[i]
        group = groups[i]
        if scaler_fit is not None:scaler_fit_group = scaler_fit_groups[i]
        predicted_scores = predict(group, model,scaler_fit=scaler_fit_group)
        matrices_for_splits = pd.concat([get_classes_n_build_matrics_df(group[['label']], predicted_scores, x / 100) for x in range(100)])
        optimal_split = matrices_for_splits[matrices_for_splits['f1'] == matrices_for_splits['f1'].max()]
        optimal_split_value = optimal_split.at[0,'split_value']
        optimal_split_value = optimal_split_value[0] if isinstance(optimal_split_value,np.ndarray) else optimal_split_value
        splits += [optimal_split_value]
    return splits

def predict_n_classify_groups(cluster_id, df,optimal_split_values,scaler_fit = None):
    models = get_cluster(cluster_id, len(fare_groups_info))
    groups = get_fare_groups(df)
    if scaler_fit is not None:scaler_fit_groups = get_fare_groups(scaler_fit)
    for i in range(len(fare_groups_info)):
        model = models[i]
        group = groups[i]
        if scaler_fit is not None:scaler_fit_group = scaler_fit_groups[i]
        predicted_scores = predict(group, model,scaler_fit=scaler_fit_group)
        group['class'] = get_classes(group[['label']], predicted_scores, optimal_split_values[i])
        group['predicted_scores'] = predicted_scores
    return groups

def pd_vstack(dfs):
    big_list = None
    for i in dfs:
        if big_list is None: big_list = i
        else:big_list = big_list.append(i, ignore_index=True)
    return big_list
def cluster_test(df,cluster_id,scaler_fit = None):
    optimal_split_values = predict_groups_and_calc_optimal_split(cluster_id,df,scaler_fit)
    result_dfs = predict_n_classify_groups(cluster_id, df,optimal_split_values,scaler_fit)
    report = pd_vstack(result_dfs)
    matrices = calc_matrices(report)
    return report,matrices

def auto_cluster_train(cluster_id,
                       df,
                       general_model_name = "",
                       no_base_model = False,
                       base_model_epochs = 10,
                       cluster_model_epochs = 20,
                       skip_base_model_training = False,
                       sleep_time = 30,
                       new_optimizer = False):
    group_index = None
    if len(sys.argv) >= 2:
        group_index = int(sys.argv[1])


    if group_index is None:
        if not (skip_base_model_training or no_base_model):
            model = build_model()
            train_save(df, model, base_model_epochs, general_model_name)
        for i in range(len(fare_groups_info)):
            print('Starting Model ' + str(i))
            Popen("python greed.py " + str(i), creationflags=CREATE_NEW_CONSOLE)
            time.sleep(sleep_time)
    else:
        groups = get_fare_groups(df)
        train_cluster(general_model_name, groups, cluster_model_epochs, cluster_id, group_number=group_index,
                      new_optimizer=new_optimizer,no_base_model = no_base_model)

    input("Press Enter to continue...")
#-----------------------------------------------------------------------------------------------------------------------

df = load_train_train()
print(42)
#auto_cluster_train('friedrichengels9',cluster_model_epochs=10000,skip_base_model_training=False,df=df,no_base_model=True,sleep_time=40)
print(cluster_test(load_train_test(),"friedrichengels9",load_train_train()))