import numpy as np
import pandas as pd
import os

seq_len = 30
pred_len = 5

EXP_NO = 'multitask_seq' + str(seq_len) + '_pred' + str(pred_len)
if not os.path.exists('data/multitask/' + EXP_NO):
    os.makedirs('data/multitask/' + EXP_NO)

#1d data
data = np.load('data/data_priceOnly.npy')
#train 240days, test 181days
train = data[:240, :]
test = data[240:, :]

train_past = []
train_present = []
train_future = []
for day in range(240):
    for pred_time in range(31, 350, 1):
        past = train[day][(pred_time-seq_len):pred_time]
        present = train[day][pred_time]
        future = train[day][(pred_time+1):(pred_time+1+pred_len)]

        train_past.append(past)
        train_present.append(present)
        train_future.append(future)
print np.array(train_past).shape #76560, 30
np.save('data/multitask/' + EXP_NO + '/train_past.npy', np.array(train_past))
np.save('data/multitask/' + EXP_NO + '/train_present.npy', np.array(train_present))
np.save('data/multitask/' + EXP_NO + '/train_future.npy', np.array(train_future))


test_past = []
test_present = []
test_future = []
for day in range(181):
    for pred_time in range(31, 350, 1):
        past = test[day][(pred_time-seq_len):pred_time]
        # present = test[day][pred_time]
        future = test[day][(pred_time+1):(pred_time+1+pred_len)]

        test_past.append(past)
        # test_present.append(present)
        test_future.append(future)
print np.array(test_past).shape #57739, 30
np.save('data/multitask/' + EXP_NO + '/test_past.npy', np.array(test_past))
np.save('data/multitask/' + EXP_NO + '/test_future.npy', np.array(test_future))

for day in range(181):
    for pred_time in range(31, 368, 1):
        present = test[day][pred_time]

        test_present.append(present)

np.save('data/multitask/' + EXP_NO + '/test_present.npy', np.array(test_present))
