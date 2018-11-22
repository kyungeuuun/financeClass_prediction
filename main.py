import os
import tensorflow as tf
from utils_model import *

def main():
    num_trials = 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    sess = tf.Session()
    K.set_session(sess)

    # EXP_NO = '1d___f_5min_seq_30min'
    # trainX = np.load('data/1d_train_x.npy')
    # trainY = np.load('data/1d_train_y.npy')
    # valX = np.load('data/1d_val_x.npy')
    # valY = np.load('data/1d_val_y.npy')
    # testX = np.load('data/1d_test_x.npy')
    # testY = np.load('data/1d_test_y.npy')

    # EXP_NO = '2d_test'
    # trainX = np.load('data/all_train_x.npy')
    # trainY = np.load('data/all_train_y.npy')
    # valX = np.load('data/all_val_x.npy')
    # valY = np.load('data/all_val_y.npy')
    # testX = np.load('data/all_test_x.npy')
    # testY = np.load('data/all_test_y.npy')

    EXP_NO = 'multitask_seq30_pred5'
    trainX = np.load('data/multitask/' + EXP_NO + '/train_past.npy')
    trainY = np.load('data/multitask/' + EXP_NO + '/train_future.npy')
    valX = np.load('data/multitask/' + EXP_NO + '/train_past.npy')
    valY = np.load('data/multitask/' + EXP_NO + '/train_future.npy')
    testX = np.load('data/multitask/' + EXP_NO + '/test_past.npy')
    testY = np.load('data/multitask/' + EXP_NO + '/test_future.npy')

    print trainX.shape
    print trainY.shape

    seq_len = 30
    pred_len = 5
    n_features = 1
    n_lstm_layers = 1
    lstm_units = 128
    batch_size = 16
    act_lstm = 'tanh'
    dropout_on = 0
    n_fc_layers = 1
    lstm_fc_units = 32
    last_fc_units = 0
    opt_learningrate = 1e-3
    loss_func = 'mean_squared_error'
    max_epoch = 50

    for trial_no in range(num_trials):
        multitask_model(EXP_NO, pred_len, trainX, trainY, valX, valY, testX, testY, seq_len, n_features,
                        n_lstm_layers, lstm_units, batch_size, act_lstm, dropout_on, n_fc_layers, lstm_fc_units, last_fc_units,
                        opt_learningrate, loss_func, max_epoch)

gpu_id = 1
# if not os.path.exists('/home/keun/PycharmProjects/utm/codes2/log/' + EXP_):
#     os.makedirs('/home/keun/PycharmProjects/utm/codes2/log/' + EXP_)
if __name__ == '__main__':
    main()