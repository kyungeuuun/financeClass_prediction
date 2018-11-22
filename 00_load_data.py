import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/min_data.csv')
# print data.columns
# Index([u'date', u'entry_time', u'entry_price', u'lob_num', u'lob_quant', u'foreign_KOSPI', u'foreign_fut', u'foreign_call', u'foreign_put',
#        u'kq_idx', u'kospi200_idx', u'sec_idx', u'cha_prog', u'bicha_prog', u'k_bond', u'dollar_fut', u'ins_KOSPI', u'ins_fut', u'long_exit_time',
#        u'long_exit_price', u'short_exit_time', u'short_exit_price'], dtype='object')

data.columns = range(22)
# data = data[[0, 1, 2]]
# data.columns = ['date', 'time', 'price']
data = data[range(18)]
data.columns = ['date', 'time', 'price', 'lob_num', 'lob_quant', 'foreign_KOSPI', 'foreign_fut', 'foreign_call', 'foreign_put',
                'kq_idx', 'kospi200_idx', 'sec_idx', 'cha_prog', 'bicha_prog', 'k_bond', 'dollar_fut', 'ins_KOSPI', 'ins_fut']

data = data[data['date'] > 1170102].reset_index(drop=True)
timerange = data.groupby(['date']).agg(['min', 'max'])['time']
# print timerange[timerange['min'] > 902]
# print timerange[timerange['max'] < 1509]
# #           min   max
# # date
# # 1171116  1002  1510
# # 1171123  1002  1510
# # 1180102  1002  1509
# #          min  max
# # date
# # 1181001  902  902

data = data[~data['date'].isin([1171116, 1171123, 1180102, 1181001])]
data['date'] = data['date'] - 1000000 + 20000000
year = (data['date']/10000).astype(int)
month = (data['date']/100).astype(int) - year*100
day = data['date'] % 100
hour = (data['time'] / 100).astype(int)
minute = data['time'] % 100

timestamp = {'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute}
data['timestamp'] = pd.to_datetime(timestamp)
data = data.sort_values('timestamp').reset_index(drop=True)
# count = pd.DataFrame(data[['date', 'timestamp']].groupby('date').agg(['count']))
# count.to_csv('dt.csv')
# print count[count]
# print data[data['date'] > 20180000].index #88320~
# print len(np.unique(data[data['date'] > 20180000]['date'].values)) #test: 181days
# print len(np.unique(data[data['date'] < 20180000]['date'].values)) #train: 240days
# plt.plot(data['timestamp'], data['price'])
# plt.show()

# #basic statistics
# data['price'] = data['price'] * 25
#
# stat = data.groupby('date').agg(['mean', 'std'])['price']
# stat['date'] = stat.index
# year = (stat['date']/10000).astype(int)
# month = (stat['date']/100).astype(int) - year*100
# day = stat['date'] % 100
# stat['timestamp'] = pd.to_datetime({'year': year, 'month': month, 'day': day})
#
# # plt.plot(stat['timestamp'], stat['mean'], c='r')
# # plt.plot(stat['timestamp'], stat['mean'] + stat['std'], linestyle='dashed', c='k', alpha=0.4)
# # plt.plot(stat['timestamp'], stat['mean'] - stat['std'], linestyle='dashed', c='k', alpha=0.4)
# # plt.show()
#
# # #problem setting
# # p = data[data['date'] == 20170103]
# # plt.plot(p['timestamp'], p['price'])
# # plt.show()

np.save('data/data_priceOnly.npy', data['price'].values.reshape([-1, 368]))
print data['price'].values.reshape([-1, 368]).shape # 421, 368 (421=#days, 368=#times in day)

result = []
for day in range(421):
    daily = []
    for t in range(368): #range(data.shape[0]):
        row = 368*day + t
        daily.append(data.loc[row].values[2:18])
    result.append(daily)

print np.array(result).shape # 421, 368, 16
np.save('data/data_all.npy', np.array(result))