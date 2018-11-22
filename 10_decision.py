import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP_NO = '1d___f_5min_seq_30min'

truth = np.load('data/1d_test_y.npy')
result = np.load('log/' + EXP_NO + '_testResult.npy')

data = np.load('data/data_priceOnly.npy')
#train 240days, test 181days
train = data[:240, :]
test = data[240:, :]

pred_y = []
for day in range(181):
    for pred_time in range(58, 298, 5):
        y = test[day][pred_time]

        pred_y.append(y)

# plt.plot(truth, c='b')
# plt.plot(result, c='r')
# plt.plot(pred_y, c='k', alpha=0.7)
# plt.show()

# plt.plot(pred_y - result)
# plt.show()

pred_y = np.array(pred_y).reshape([-1, 48])
truth = np.array(truth).reshape([-1, 48])
result = np.array(result).reshape([-1, 48])
# print pred_y.shape
# timeRange = pd.date_range(start='10:00', end='14:00', freq='5T', closed='left').strftime('%H:%M')
#
# for day in range(181):
#     plt.figure()
#     plt.plot(timeRange, truth[day]*25, c='b', label='Truth(t+30)')
#     plt.plot(timeRange, result[day]*25, c='r', label='Predicted(t+30)')
#     plt.plot(timeRange, pred_y[day]*25, c='k', label='Truth(t)')
#     plt.legend(loc = 'upper right')
#     plt.xticks(pd.date_range(start='10:00', end='14:00', freq='1H').strftime('%H:%M'))
#     plt.title('Test Result at Day: %i' %day)
#     plt.ylabel('Price(x10,000won)')
#     plt.xlabel('Time in day')
#     plt.savefig('plots/1d_test/' + str(day) + '.png')
#     plt.clf()
#     # plt.show()

# for day in range(1): #range(181):
#     pred_y[day]

decisionTable = pd.DataFrame(columns = ('day', 'present_price', 'predicted_price', 'future_price', 'decision'))
i = 0
criteria_buy = 0.5
criteria_sell = -0.5

for day in range(181):
    for time in range(48):
        present_price = pred_y[day][time]
        predicted_price = result[day][time]
        future_price = truth[day][time]

        expected_benefit = predicted_price - present_price
        if expected_benefit > criteria_buy:
            decision = -1 #'buy'
        elif expected_benefit < criteria_sell:
            decision = 1 #'sell'
        else:
            decision = 0 #'hold'

        decisionTable.loc[i] = [day, present_price, predicted_price, future_price, decision]
        i += 1

commission = 0.03
decisionInterval = 6

print decisionTable.head()
decisionTable.to_csv('log/' + EXP_NO + '_decision.csv')

# decisionTable = pd.read_csv('log/1d_test_decision.csv')
#
# real_decision = []
# for day in range(3): #range(181):
#     dt = decisionTable[decisionTable['day'] == day].reset_index(drop=True)
#     real_decision_at_t = []
#
#     for i in range(dt.shape[0]):
#         if dt['decision'].values[i] == 0:
#             real_decision_at_t += [0]
#         elif dt['decision'].values[i] == -1:
#              dt['decision'][i:(i+6)]
#
#
#     # budget = 0.0
#     # for i in range(dt.shape[0]):
#     #     dt_at_t = dt.loc[i]
#     #     if dt_at_t['decision'] != 0:
#     #         budget += dt_at_t['decision'] * dt_at_t['present_price'] - commission
#
#
#
#
# # print sum(decisionTable['decision'] == 'sell')
# # print sum(decisionTable['decision'] == 'buy')
# # print sum(decisionTable['decision'] == 'hold')
# # print decisionTable

#