import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from dateutil import parser
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense
import keras
from keras import backend as K

data=None
columns = [
    'tripid',
    'additional_fare',
    'duration',
    'meter_waiting',
    'meter_waiting_fare',
    'meter_waiting_till_pickup',
    'pickup_time',
    'drop_time',
    'pick_lat',
    'pick_lon',
    'drop_lat',
    'drop_lon',
    'fare',
    'label',
    'distance',
    'avg_time'
]


def cg_label():
    global data
    column = columns.index('label')
    for y in range(data.shape[0]):
        data[y][column] = 1 if data[y][column] == 'correct' else 0


def rm_nan():
    global data
    c = 0
    y = 0

    fs = 0

    column = columns.index('label')
    while y < data.shape[0]:
        jj = np.random.rand()<delpos
        for x in range(data.shape[1]):
            if (not (isinstance(data[y][x], str)) and math.isnan(data[y][x])) or (data[y][column]=='correct' and  jj):
                c += 1
                data = np.delete(data, y, axis=0)
                y -= 1
                break
        y += 1
    print(fs)
    return c


def mini(st):
    d = parser.parse(st)
    return d.time().hour * 60 + d.time().minute


def add_dist():
    global data, columns
    data = np.append(data, np.zeros((len(data), 1), dtype=np.float32), axis=1)
    column = data.shape[1] - 1
    col_pick_lat = columns.index('pick_lat')
    col_pick_lon = columns.index('pick_lon')
    col_drop_lat = columns.index('drop_lat')
    col_drop_lon = columns.index('drop_lon')

    for y in range(data.shape[0]):
        d = (data[y][col_drop_lat] - data[y][col_pick_lat]) ** 2 + (data[y][col_drop_lon] - data[y][col_pick_lon]) ** 2
        data[y][column] = np.sqrt(d)


def add_avg_time():
    global data, columns
    data = np.append(data, np.zeros((len(data), 1), dtype=np.float32), axis=1)
    column = data.shape[1] - 1

    col_pick_tim = columns.index('pickup_time')
    col_drop_tim = columns.index('drop_time')

    for y in range(data.shape[0]):
        data[y][column] = (mini(data[y][col_pick_tim]) + mini(data[y][col_drop_tim])) / 2


def gdat(name,scaler,fitscaler=True,label = True,shuffle = True,rm=True):
    global data
    data = pd.read_csv(name).values
    if rm:rm_nan()
    if label:cg_label()
    add_dist()
    add_avg_time()

    if shuffle:np.random.shuffle(data)

    inp_colomns = [
        #                     'tripid',
        'additional_fare',
        'duration',
        #'meter_waiting',
        'meter_waiting_fare',
        #'meter_waiting_till_pickup',
        #                     'pickup_time',
        #                     'drop_time',
        #                     'pick_lat',
        #                     'pick_lon',
        #                     'drop_lat',
        #                     'drop_lon',
        'fare',
        #                     'label',
        'distance',
        'avg_time'
    ]


    inp = data[:, [columns.index(x) for x in inp_colomns]]
    #out = data[:, [columns.index(x) for x in ['label']]]
    out = [x[columns.index('label')] for x in data] if label else None

    if fitscaler:inp = scaler.fit_transform(inp)
    else: inp = scaler.transform(inp)

    return inp,out


p=0.0
delpos = .0

scaler = MinMaxScaler()
all = gdat('test.csv',scaler,shuffle=False,label=False,rm=False)

inp,out = all

inp2 = inp[:int(len(inp)*p)]
inp1 = inp[int(len(inp)*p):]
#out2 = out[:int(len(out)*p)]
#out1 = out[int(len(out)*p):]



# classifier = LogisticRegression().fit(inp1, out1)
# print(classifier.score(inp2, out2))


model = load_model("./finaleee")
# model = Sequential()
# model.add(Dense(3, input_dim=inp.shape[1], activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
qq = model.predict(inp)

#model.fit(inp1,out1,epochs=3000)

#model.save("./mod")



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def eval(o=.5):
    rslts = qq>o
    expctd = out >.5
    (TP, FP, TN, FN) = perf_measure(expctd.reshape(len(expctd)), rslts.reshape(len(rslts)))
    Precision = TP / (TP + FP)
    Recall = TP/(TP+FN)
    f1 = 2 * (Recall * Precision) / (Recall + Precision)
    return f1