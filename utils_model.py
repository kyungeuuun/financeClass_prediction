import numpy as np
import pandas as pd
import time as tt
from keras import activations
from keras.engine.topology import Layer
from keras.layers import LSTM, InputLayer, Dense, Input, Flatten, concatenate, Reshape, MaxPooling2D, Activation, Dropout, MaxPooling3D, Conv3D
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def nanmean(x):
    return tf.reduce_mean(tf.boolean_mask(x, tf.logical_not(tf.is_inf(x))), axis = -1)

def mean_absolute_percentage_error(y_true, y_pred):
    return nanmean(K.abs(y_true - y_pred) / y_true)

def mean_absolute_error(y_true, y_pred):
    return nanmean(K.abs(y_true - y_pred))

def mean_root_squared_error(y_true, y_pred):
    return tf.sqrt(nanmean(K.square(y_true - y_pred)))

def perfMetrics(err, truth):
    mae = np.mean(np.abs(err))
    mape = np.true_divide(np.abs(err), truth)
    mape = mape[~np.isnan(mape)]
    mape = np.mean(mape)
    rmse = np.sqrt(np.mean(np.square(err)))

    return (mape, mae, rmse)

class EpochHistory(Callback):
    def __init__(self, trainX, trainY, valX, valY, testX, testY, log_filepath, testResult_filepath):
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        self.log_filepath = log_filepath
        self.testResult_filepath = testResult_filepath

    def on_train_begin(self, logs={}):
        self.perf = pd.DataFrame(columns = ['epoch', 'train_mape', 'train_mae', 'train_rmse', 'val_mape', 'val_mae', 'val_rmse',
                                            'test_mape', 'test_mae', 'test_rmse', 'train_time'])
        self.testResult = []
        self.st = tt.time()

    def on_epoch_end(self, epoch, logs={}):
        train_predicted = np.squeeze(self.model.predict(self.trainX))
        val_predicted = np.squeeze(self.model.predict(self.valX))
        test_predicted = np.squeeze(self.model.predict(self.testX))

        train = perfMetrics(train_predicted - self.trainY, self.trainY)
        val = perfMetrics(val_predicted - self.valY, self.valY)
        test = perfMetrics(test_predicted - self.testY, self.testY)

        self.perf.loc[epoch] = [epoch, train[0], train[1], train[2], val[0], val[1], val[2], test[0], test[1], test[2], tt.time() - self.st]
        print('### EPOCH %i / TIME %.1f ### train %.3f %.3f %.3f // validation %.3f %.3f %.3f // test %.3f %.3f %.3f' %(
            epoch, tt.time() - self.st, train[0], train[1], train[2], val[0], val[1], val[2], test[0], test[1], test[2]))

        return

    def on_train_end(self, logs={}):
        self.testResult = np.squeeze(self.model.predict(self.testX))

        np.save('/home/keun/PycharmProjects/financeClass/log/' + str(self.testResult_filepath) + '.npy', self.testResult)
        self.perf.to_csv('/home/keun/PycharmProjects/financeClass/log/' + str(self.log_filepath) + '.csv')


# def repeated_lstm(input, n_lstm_layers, seq_len, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units):
#     lstm = Reshape(target_shape=(seq_len, -1), name='reshape')(input)
#
#     if n_lstm_layers == 1:
#         lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=False, name='lstm')(lstm)
#         lstm = Activation(act_lstm)(lstm)
#     else:
#         lstm = LSTM(units=lstm_units, batch_input_shape=(batch_size, seq_len, lstm_units), return_sequences=True)(lstm)
#         lstm = Activation(act_lstm)(lstm)
#         lstm = LSTM(units=lstm_units, return_sequences=False)(lstm)
#         lstm = Activation(act_lstm)(lstm)
#
#     if dropout_on == 1:
#         res = Dropout(0.2)(lstm)
#     else:
#         res = lstm
#     res = Dense(units=lstm_fc_units, name='dense_1')(res)
#
#     if n_fc_layers == 2:
#         res = Dense(units=last_fc_units)(res)
#
#     return res

def lstm_model(EXP_NO, trainX, trainY, valX, valY, testX, testY, seq_len, n_features,
               n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
               opt_learningrate, loss_func, max_epoch):

    if n_features == 1:
        input = Input(shape=(seq_len, ), name='lstm_input')
    else:
        input = Input(shape=(seq_len, n_features, ), name='lstm_input')

    # res = repeated_lstm(input, n_lstm_layers, seq_len, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units)
    lstm = Reshape(target_shape=(seq_len, -1), name='reshape')(input)

    lstm = LSTM(units=lstm_units, input_shape=(seq_len, n_features), return_sequences=False, name='lstm')(lstm)
    lstm = Activation(act_lstm)(lstm)

    res = Dense(units=lstm_fc_units, name='dense_1')(lstm)
    target = Dense(units=1, name='dense_2')(res)

    model = Model(inputs = [input], outputs = target)

    print model.summary()

    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(trainX=trainX, trainY=trainY, valX=valX, valY=valY, testX=testX, testY=testY,
                           log_filepath=str(EXP_NO), testResult_filepath=str(EXP_NO) + '_testResult')

    model.fit([trainX], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    return


def multitask_model(EXP_NO, pred_len, trainX, trainY, valX, valY, testX, testY, seq_len, n_features,
                    n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
                    opt_learningrate, loss_func, max_epoch, normalized = False):

    if n_features == 1:
        input = Input(shape=(seq_len, ), name='lstm_input')
    else:
        input = Input(shape=(seq_len, n_features, ), name='lstm_input')

    if normalized:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        input = scaler.fit(input)

    lstm = Reshape(target_shape=(seq_len, -1), name='reshape')(input)

    lstm = LSTM(units=lstm_units, input_shape=(seq_len, n_features), return_sequences=False, name='lstm')(lstm)
    lstm = Activation(act_lstm)(lstm)

    res = Dense(units=lstm_fc_units, name='dense_1')(lstm)
    target = Dense(units=pred_len, name='dense_2')(res)

    model = Model(inputs = [input], outputs = target)

    print model.summary()

    opt = Adam(lr=opt_learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

    model.compile(loss=loss_func, optimizer=opt)
    history = EpochHistory(trainX=trainX, trainY=trainY, valX=valX, valY=valY, testX=testX, testY=testY,
                           log_filepath=str(EXP_NO), testResult_filepath=str(EXP_NO) + '_testResult')

    model.fit([trainX], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1, callbacks=[history], verbose=0, shuffle=True)

    return
