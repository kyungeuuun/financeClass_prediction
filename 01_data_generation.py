import numpy as np
import pandas as pd

dailyTimeRange = pd.date_range(start='09:02', end='15:09', freq='1T').strftime('%H:%M')
# print len(dailyTimeRange) #368

# print np.where(dailyTimeRange == '10:00')[0] #first prediction at 58
# print np.where(dailyTimeRange == '14:00')[0] #last prediction at 298
# print train[0].shape

# print np.where(dailyTimeRange == '09:35')[0] #first prediction at 33
# print np.where(dailyTimeRange == '15:00')[0] #last prediction at 358


forecasting_horizon = 5
seq_len = 30

#2d data
data = np.load('data/data_all.npy')
y_data = np.load('data/data_priceOnly.npy')
train = data[:240, :, :]
# print train.shape #(240, 368, 16)
y_for_train = y_data[:240]
test = data[240:, :, :]
y_for_test = y_data[240:]
# print train

train_x = []
train_y = []
for day in range(240):
    for pred_time in range(33, 358, 5):
        y = y_for_train[day][pred_time + forecasting_horizon]
        x = train[day, (pred_time - seq_len):pred_time, :]

        train_x.append(np.array(x))
        train_y.append(y)

print np.array(train_x).shape #15600
print np.array(train_y).shape

np.random.seed(seed=1121)
val_idx = np.random.choice(range(15600), 1600, replace=False)
val_x = np.array(train_x)[val_idx, :, :]
val_y = np.array(train_y)[val_idx]

train_idx = [i for i in range(15600) if i not in val_idx]
train_x = np.array(train_x)[train_idx, :, :]
train_y = np.array(train_y)[train_idx]

np.save('data/all_train_x.npy', np.array(train_x))
np.save('data/all_train_y.npy', np.array(train_y))
np.save('data/all_val_x.npy', np.array(val_x))
np.save('data/all_val_y.npy', np.array(val_y))

test_x = []
test_y = []
for day in range(181):
    for pred_time in range(33, 358, 5):
        y = y_for_test[day][pred_time + forecasting_horizon]
        x = test[day][(pred_time - seq_len): pred_time, :]

        test_x.append(np.array(x))
        test_y.append(y)

np.save('data/all_test_x.npy', np.array(test_x))
np.save('data/all_test_y.npy', np.array(test_y))

print train_x.shape #14000
print val_x.shape #1600
print np.array(test_x).shape #11765


print train_y.shape #14000
print val_y.shape #1600
print np.array(test_y).shape #11765



# #1d data
# data = np.load('data/data_priceOnly.npy')
# #train 240days, test 181days
# train = data[:240, :]
# test = data[240:, :]
#
# train_x = []
# train_y = []
# for day in range(240):
#     for pred_time in range(58, 298, 5):
#         y = train[day][pred_time + forecasting_horizon]
#         x = train[day][(pred_time - seq_len) : pred_time]
#
#         train_x.append(x)
#         train_y.append(y)
#
# np.random.seed(seed=1121)
# val_idx = np.random.choice(range(11520), 1520, replace=False)
# val_x = np.array(train_x)[val_idx]
# val_y = np.array(train_y)[val_idx].tolist()
#
# train_idx = [i for i in range(11520) if i not in val_idx]
# train_x = np.array(train_x)[train_idx]
# train_y = np.array(train_y)[train_idx].tolist()
#
# np.save('data/1d_train_x.npy', np.array(train_x))
# np.save('data/1d_train_y.npy', np.array(train_y))
# np.save('data/1d_val_x.npy', np.array(val_x))
# np.save('data/1d_val_y.npy', np.array(val_y))
#
# test_x = []
# test_y = []
# for day in range(181):
#     for pred_time in range(58, 298, 5):
#         y = test[day][pred_time + forecasting_horizon]
#         x = test[day][(pred_time - seq_len): pred_time]
#
#         test_x.append(x)
#         test_y.append(y)
#
# np.save('data/1d_test_x.npy', np.array(test_x))
# np.save('data/1d_test_y.npy', np.array(test_y))