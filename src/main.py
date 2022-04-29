import matplotlib.pylab as pylab
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import tqdm
import pandas as pd
import numpy as np
!pip install - qq tensorflow == 2.1.0
!pip install - qq - -upgrade wandb

plt.rcParams['figure.figsize'] = 12, 8


### tensorflow 2 ###

# Grid search
# import talos

df = pd.read_excel('./data/tax-sales.csv', index_col=0)
category = 'Hotel'
print(df.shape)
df.tail()
countyNames = df["region"].unique()
countySplit = int(len(countyNames)*.8)
trainCounties = countyNames[10:]
testCounties = countyNames[0:10]
print('Total counties for', category, "category :", len(countyNames))
print('Train counties for', category, "category :", len(trainCounties))
print('Test counties for', category, "category :", len(testCounties))

df
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(
    df[['hurricane', 'observed', 'residual'	, 'seasonal'	, 'trend']])
df[['hurricane', 'observed', 'residual'	, 'seasonal'	, 'trend']] = scaler.transform(
    df[['hurricane', 'observed', 'residual'	, 'seasonal'	, 'trend']])
df

testCounties = ['Lee', 'Collier', 'Hardee', 'Sarasota', 'Orange', 'Palm Beach']
trainCounties = ['Alachua', 'Bay', 'Brevard', 'Broward', 'Citrus', 'Clay',
                 'DeSoto', 'Dixie', 'Duval', 'Escambia', 'Franklin',
                 'Gadsden', 'Gulf',  'Hendry', 'Hernando', 'Highlands',
                 'Hillsborough', 'IndianRiver', 'Jefferson', 'Lake',  'Leon',
                 'Levy', 'Manatee', 'Marion', 'Martin', 'Miami-Dade', 'Monroe',
                 'Nassau', 'Okaloosa',  'Osceola',
                 'Pinellas', 'Polk', 'Putnam', 'St. Johns', 'St. Lucie',
                 'Santa Rosa',  'Seminole', 'Sumter', 'Suwannee',
                 'Volusia', 'Wakulla', 'Walton']


### CREATE GENERATOR FOR LSTM WINDOWS AND LABELS ###
lossmetric = 'mean_absolute_error'
sequence_length = 8
dropae = 0.1


def gen_sequence(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]


### CREATE TRAIN/TEST PRICE DATA ###
X_train_c, X_train_o = [], []
X_test_c, X_test_o = [], []
X_other_train_c, X_other_train_o = [], []
X_other_test_c, X_other_test_o = [], []

for county in df["region"].unique():

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                 sequence_length, ['observed']):
        if county in trainCounties:
            X_train_c.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                 sequence_length, ['observed']):
        if county in trainCounties:
            X_train_o.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                 sequence_length, ['observed']):
        if county in testCounties:
            X_test_c.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                 sequence_length, ['observed']):
        if county in testCounties:
            X_test_o.append(sequence)


X_train_c, X_train_o = np.asarray(X_train_c), np.asarray(X_train_o)
X_test_c, X_test_o = np.asarray(X_test_c), np.asarray(X_test_o)
y_train_c, y_train_o = [], []
y_test_c, y_test_o = [], []
y_other_train_c, y_other_train_o = [], []
y_other_test_c, y_other_test_o = [], []

for county in df["region"].unique():

    for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Urban")],
                               sequence_length, ['observed']):
        if county in trainCounties:
            y_train_c.append(sequence)

    for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Rural")],
                               sequence_length, ['observed']):
        if county in trainCounties:
            y_train_o.append(sequence)

    for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Urban")],
                               sequence_length, ['observed']):
        if county in testCounties:
            y_test_c.append(sequence)

    for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Rural")],
                               sequence_length, ['observed']):
        if county in testCounties:
            y_test_o.append(sequence)


y_train_c, y_train_o = np.asarray(y_train_c), np.asarray(y_train_o)
y_test_c, y_test_o = np.asarray(y_test_c), np.asarray(y_test_o)
df['year'].unique()
splitTest = 197
customXLabel = []
for j in range(len(df['year'].unique())-1):
    j = j*12
    customXLabel.append(j)
customXLabel
### CONCATENATE TRAIN/TEST DATA AND LABEL ###
XUrban = np.concatenate([X_train_c, X_test_c], axis=0)
XRural = np.concatenate([X_train_o, X_test_o], axis=0)

yUrban = np.concatenate([y_train_c, y_test_c], axis=0)
yRural = np.concatenate([y_train_o, y_test_o], axis=0)


print('Urban: ', XUrban.shape, ' - Rural ', XRural.shape)
### CREATE TRAIN/TEST EXTERNAL FEATURES ###
col = ['residual', "seasonal", 'trend', 'hurricane']

f_train_c, f_train_o = [], []
f_test_c, f_test_o = [], []
f_other_train_c, f_other_train_o = [], []
f_other_test_c, f_other_test_o = [], []


for county in df["region"].unique():

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                 sequence_length, col):
        if county in trainCounties:
            f_train_c.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                 sequence_length, col):
        if county in trainCounties:
            f_train_o.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                 sequence_length, col):
        if county in testCounties:
            f_test_c.append(sequence)

    for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                 sequence_length, col):
        if county in testCounties:
            f_test_o.append(sequence)


f_train_c, f_train_o = np.asarray(f_train_c), np.asarray(f_train_o)
f_test_c, f_test_o = np.asarray(f_test_c), np.asarray(f_test_o)
### CONCATENATE TRAIN/TEST EXTERNAL FEATURES ###
F = np.concatenate([f_train_c, f_train_o, f_test_c, f_test_o], axis=0)
FUrban = np.concatenate([f_train_c, f_test_c], axis=0)
FRural = np.concatenate([f_train_o, f_test_o], axis=0)

print(FUrban.shape, FRural.shape)

"""# AutoEncoder Urban - Rural """

### SET SEED ###
# tf.random.set_seed(47)
os.environ['PYTHONHASHSEED'] = str(47)
np.random.seed(47)
random.seed(47)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

### LSTM AUTOENCODER ###
inputsUrbanAE = Input(shape=(sequence_length, 1))
encodedUrbanAE = GRU(128, return_sequences=True, dropout=dropae)(
    inputsUrbanAE, training=True)
decodedUrbanAE = GRU(32, return_sequences=True, dropout=dropae)(
    encodedUrbanAE, training=True)
outputUrbanAE = TimeDistributed(Dense(1))(decodedUrbanAE)

sequenceUrbanAE = Model(inputsUrbanAE, outputUrbanAE)
sequenceUrbanAE.compile(
    optimizer='adam', loss=lossmetric, metrics=[lossmetric])

### TRAIN AUTOENCODER ###
sequenceUrbanAE.fit(XUrban[:len(X_train_c)], XUrban[:len(X_train_c)],
                    # validation_data=(XUrban[len(X_train_c):], XUrban[len(X_train_c):]),
                    batch_size=128, epochs=30, verbose=2, shuffle=False)
os.environ['PYTHONHASHSEED'] = str(47)
np.random.seed(47)
random.seed(47)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

### LSTM AUTOENCODER ###
inputsRuralAE = Input(shape=(sequence_length, 1))
encodedRuralAE = GRU(128, return_sequences=True, dropout=dropae)(
    inputsRuralAE, training=True)
decodedRuralAE = GRU(32, return_sequences=True, dropout=dropae)(
    encodedRuralAE, training=True)
outputRuralAE = TimeDistributed(Dense(1))(decodedRuralAE)

sequenceRuralAE = Model(inputsRuralAE, outputRuralAE)
sequenceRuralAE.compile(
    optimizer='adam', loss=lossmetric, metrics=[lossmetric])

### TRAIN AUTOENCODER ###
sequenceRuralAE.fit(XRural[:len(X_train_o)], XRural[:len(X_train_o)],
                    batch_size=128, epochs=5, verbose=2, shuffle=False)

"""# GRU after AutoEncoder Urban & Rural"""

os.environ['PYTHONHASHSEED'] = str(47)
np.random.seed(47)
random.seed(47)

dropOut = 0.3
epochs = 400
lstmcell1Urban = 256
lstmcell2Urban = 128
dense1Urban = 64

lstmcell1Rural = 128
lstmcell2Rural = 32
dense1Rural = 32


encoderUrban = Model(inputsUrbanAE, encodedUrbanAE)
XXUrban = encoderUrban.predict(XUrban)
XXFUrban = np.concatenate([XXUrban, FUrban], axis=2)
XXFUrban.shape
X_train1Urban, X_test1Urban = XXFUrban[:len(
    X_train_c)], XXFUrban[len(X_train_c):]
y_train1Urban, y_test1Urban = yUrban[:len(y_train_c)], yUrban[len(y_train_c):]
scaler1Urban = StandardScaler()
X_train1Urban = scaler1Urban.fit_transform(
    X_train1Urban.reshape(-1, XXFUrban.shape[-1])).reshape(-1, sequence_length, XXFUrban.shape[-1])
X_test1Urban = scaler1Urban.transform(
    X_test1Urban.reshape(-1, XXFUrban.shape[-1])).reshape(-1, sequence_length, XXFUrban.shape[-1])

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

### DEFINE FORECASTER ###
inputs1 = Input(shape=(X_train1Urban.shape[1], X_train1Urban.shape[2]))
lstm1 = GRU(lstmcell1Urban, return_sequences=True,
            dropout=dropOut)(inputs1, training=True)
lstm1 = GRU(lstmcell2Urban, return_sequences=False,
            dropout=dropOut)(lstm1, training=True)
dense1 = Dense(dense1Urban)(lstm1)
out1 = Dense(1)(dense1)

model1Urban = Model(inputs1, out1)
model1Urban.compile(loss=lossmetric, optimizer='adam', metrics=[lossmetric])

### FIT FORECASTER ###
historyUrban = model1Urban.fit(X_train1Urban[:4000], y_train1Urban[:4000], validation_data=(X_train1Urban[4000:], y_train1Urban[4000:]),
                               epochs=epochs, batch_size=128, verbose=2, shuffle=False)

plt.plot(historyUrban.history['loss'][10:])
plt.plot(historyUrban.history['val_loss'][10:])

plt.title('Urban-Rural model training loss 10-100')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Urban loss', 'Rural loss'], loc='upper left')
plt.show()

### FIT FORECASTER ###
# plt.plot(historyUrban.history['loss'][10:])

"""# UNSEEN DATA PREDICTION"""

### FUNCTION FOR STOCHASTIC DROPOUT FOR UNSEEN COUNTY ###


def test_other_drop1(R, typ, enc, NN, hurricane):
    if typ == 'Urban':
        X = X_test_c
        F = f_test_c
        y = y_test_c
        if hurricane == False:
            F = f_test_c_noHurricane

    elif typ == 'Rural':
        X = X_test_o
        F = f_test_o
        y = y_test_o
        if hurricane == False:
            F = f_test_o_noHurricane
    enc_pred = np.vstack(enc([X, R]))
    enc_pred = np.concatenate([enc_pred, F], axis=2)
    trans_pred = scaler1Urban.transform(
        enc_pred.reshape(-1, enc_pred.shape[-1])).reshape(-1, sequence_length, enc_pred.shape[-1])
    NN_pred = NN([trans_pred, R])

    return np.vstack(NN_pred), y


testCounties1 = ['Lee', 'Collier', 'Sarasota', 'Orange', 'Palm Beach']
testCounties1 = ['Collier']

# from sklearn.utils import check_arrays


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
          'axes.labelsize': '15',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': '18',
          'ytick.labelsize': '22',
          'font.family': 'Times new roman'}
pylab.rcParams.update(params)

lst = []
lst
lstBoxPlot = []


for j in testCounties1:
    county_name = j
    b = df[(df["region"] == j)]
    county_type = b['type'].unique()[0]
    if str(b['type'].values[0]) == 'nan':
        print('Not defined county type!')
        continue
    X_test_c, X_test_o = [], []
    y_test_c, y_test_o = [], []
    f_test_c, f_test_o = [], []
    f_test_c_noHurricane, f_test_o_noHurricane = [], []

    for county in df["region"].unique():
        for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                     sequence_length, ['observed']):
            if county == j:
                X_test_c.append(sequence)

        for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                     sequence_length, ['observed']):
            if county == j:
                X_test_o.append(sequence)
    X_test_c, X_test_o = np.asarray(X_test_c), np.asarray(X_test_o)
    ####

    for county in df["region"].unique():
        for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Urban")],
                                   sequence_length, ['observed']):
            if county == j:
                y_test_c.append(sequence)
        for sequence in gen_labels(df[(df["region"] == county) & (df["type"] == "Rural")],
                                   sequence_length, ['observed']):
            if county == j:
                y_test_o.append(sequence)
    y_test_c, y_test_o = np.asarray(y_test_c), np.asarray(y_test_o)
    ### CREATE TRAIN/TEST EXTERNAL FEATURES ###
    col = ['residual', "seasonal", 'trend', 'hurricane']
    for county in df["region"].unique():
        for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Urban")],
                                     sequence_length, col):
            if county == j:
                f_test_c.append(sequence)
        for sequence in gen_sequence(df[(df["region"] == county) & (df["type"] == "Rural")],
                                     sequence_length, col):
            if county == j:
                f_test_o.append(sequence)
    f_test_c, f_test_o = np.asarray(f_test_c), np.asarray(f_test_o)

    col = ['residual', "seasonal", 'trend', 'hurricane']
    dfnohurricane = df.copy()
    dfnohurricane['hurricane'] = 0
    for county in dfnohurricane["region"].unique():
        for sequence in gen_sequence(dfnohurricane[(dfnohurricane["region"] == county) & (dfnohurricane["type"] == "Urban")],
                                     sequence_length, col):
            if county == j:
                f_test_c_noHurricane.append(sequence)
        for sequence in gen_sequence(dfnohurricane[(dfnohurricane["region"] == county) & (dfnohurricane["type"] == "Rural")],
                                     sequence_length, col):
            if county == j:
                f_test_o_noHurricane.append(sequence)
    f_test_c_noHurricane, f_test_o_noHurricane = np.asarray(
        f_test_c_noHurricane), np.asarray(f_test_o_noHurricane)

    if county_type == "Urban":

        hurricane = True

        MAE, trueValuesList = [], []
        RMSE = []
        MAPE = []
        encUrban = K.function([encoderUrban.layers[0].input], [
                              encoderUrban.layers[-1].output])
        NNUrban = K.function([model1Urban.layers[0].input], [
                             model1Urban.layers[-1].output])
        for i in tqdm.tqdm(range(0, 100)):

            predictedValues, trueValues = test_other_drop1(
                0.2, county_type, encUrban, NNUrban, hurricane)
            MAE.append(mean_absolute_error(predictedValues, trueValues))
            RMSE.append(np.sqrt(mean_squared_error(
                predictedValues, trueValues)))
            MAPE.append(mean_absolute_percentage_error(
                predictedValues, trueValues))

            trueValuesList.append(trueValues)
            a = 'a'+str(i)
            predictedValues1 = predictedValues.reshape(1, len(predictedValues))
        print(county_name+'\t'+'Urban ', '\t',
              'MAE', np.mean(MAE), '\t', np.std(MAE))
        print(county_name+'\t'+'Urban ', '\t', 'RMSE',
              np.mean(RMSE), '\t', np.std(RMSE))
        print(county_name+'\t'+'Urban ', '\t', 'MAPE',
              np.mean(MAPE), '\t', np.std(MAPE))

        hurricane = False

        MAENoHurricane, trueValuesList = [], []
        encUrban = K.function([encoderUrban.layers[0].input], [
                              encoderUrban.layers[-1].output])
        NNUrban = K.function([model1Urban.layers[0].input], [
                             model1Urban.layers[-1].output])

        for i in tqdm.tqdm(range(0, 100)):
            predictedValuesNoHurricane, trueValues = test_other_drop1(
                0.1, county_type, encUrban, NNUrban, hurricane)
            MAENoHurricane.append(mean_absolute_error(
                predictedValuesNoHurricane, trueValues))
            trueValuesList.append(trueValues)

        print(county_name+'\t'+county_type+'\t'+'Urban no hurricane',
              '\t', np.mean(MAENoHurricane), '\t', np.std(MAENoHurricane))

    ### COMPUTE STOCHASTIC DROPOUT FOR UNSEEN COUNTY ###
    if county_type == 'Rural':

        hurricane = True

        MAE, trueValuesList = [], []
        encRural = K.function([encoderRural.layers[0].input], [
                              encoderRural.layers[-1].output])
        NNRural = K.function([model1Rural.layers[0].input], [
                             model1Rural.layers[-1].output])

        for i in tqdm.tqdm(range(0, 100)):
            predictedValues, trueValues = test_other_drop1(
                0.1, county_type, encRural, NNRural, hurricane)
            MAE.append(mean_absolute_error(predictedValues, trueValues))
            trueValuesList.append(trueValues)

        print(county_name+'\t'+county_type+'\t'+"Rural",
              '\t', np.mean(MAE), '\t', np.std(MAE))

        hurricane = False
        MAENoHurricane, trueValuesList = [], []
        encRural = K.function([encoderRural.layers[0].input], [
                              encoderRural.layers[-1].output])
        NNRural = K.function([model1Rural.layers[0].input], [
                             model1Rural.layers[-1].output])

        for i in tqdm.tqdm(range(0, 100)):
            predictedValuesNoHurricane, trueValues = test_other_drop1(
                0.1, county_type, encRural, NNRural, hurricane)
            MAENoHurricane.append(mean_absolute_error(
                predictedValuesNoHurricane, trueValues))
            trueValuesList.append(trueValues)

        print(county_name+'\t'+county_type+'\t'+'Rural no hurricane',
              '\t', np.mean(MAENoHurricane), '\t', np.std(MAENoHurricane))

    plt.rcParams['figure.figsize'] = 12, 8
    lst.append([category, dropOut, county_type, county_name, np.mean(MAENoHurricane), np.std(MAENoHurricane), np.mean(MAE), np.std(MAE)
                ])

    f, ax = plt.subplots()
    ax.plot(np.mean(np.hstack(trueValuesList).T, axis=0),
            color='black', label='Observed', marker='x')
    label1 = "AA-GRU with anomaly labeling"
    label2 = "AA-GRU without anomaly labeling"

    ax.plot(predictedValuesNoHurricane, color='blue', label=label2, marker='o')
    ax.plot(predictedValues, color='red', label=label1, marker='d')

    # ax.plot(b["hurricane"].iloc[10:].values,label = 'Anomaly Label (Hurricane 1-5)',color='orange')
    hur = b["hurricane"].iloc[12:].values
    trueVals = np.mean(np.hstack(trueValuesList).T, axis=0)

    ax.set_xticklabels(df['Date'].values[0:len(y_test_o[0:splitTest])])
    ax.set_xticks(customXLabel)
    ax.set_xticklabels(df['year'].unique()[1:])
    ax.set_xticklabels(df['Date'].values[0:len(y_test_o[0:splitTest])])
    ax.set_xticks(customXLabel)
    ax.set_xticklabels(df['year'].unique()[1:])
    ax.set_xticklabels(df['year'].unique()[1:])
    plt.ylabel('')
    plt.grid(color='k', linestyle='-', linewidth=0.4)
    plt.savefig('.//output//'+category+"_"+county_type+"_" +
                county_name+'DropOut_'+str(dropOut)+'.png')

    plt.show()
    x = 0
cols = ['Category', 'DropOut', 'type', 'countyName',
        'MAEnoHur', 'STDnoHur', 'MAE', 'STD']
excelResults = pd.DataFrame(lst, columns=cols)
districols = ['region', 'type', 'hurricane', 'Date', 'observed', 'residual',
              'true', 'predicted no hurricane', 'changes from ori', 'predicted hurricane', 'predicted changes from ori', 'changes hur to no hur', ]

excelDistri = pd.DataFrame(lstBoxPlot, columns=districols)
