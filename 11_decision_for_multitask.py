import numpy as np
import pandas as pd

EXP_NO = 'multitask_seq15_pred5'
def printBenefit(threshold, EXP_NO):
    true_present = np.load('data/multitask/' + EXP_NO + '/test_present.npy')
    pred_future = np.load('log/' + EXP_NO + '_testResult.npy')

    data = np.load('data/data_priceOnly.npy')
    #train 240days, test 181days
    test = data[240:, :]
    # print test.shape #181, 368

    result = []
    for day in range(181):
        dailyDecisionMatrix = np.zeros([368, 368], dtype=float)
        for time in range(31, 350, 1):
            i = day*319 + (time-31)
            norm = true_present[i]
            pred = pred_future[i]

            expectedBuySell = ([-norm-0.03] * 15 + pred).clip(min=threshold)
            max_expectedBuySell = max(expectedBuySell)
            expectedSellBuy = ([norm-0.03] * 15 - pred).clip(min=threshold)
            max_expectedSellBuy = max(expectedSellBuy)

            if max_expectedBuySell > max_expectedSellBuy:
                idx = np.argmax(expectedBuySell)
                idx_value = +1
            elif max_expectedBuySell < max_expectedSellBuy:
                idx = np.argmax(expectedSellBuy)
                idx_value = -1
            else:
                idx = None

            if idx is not None:
                dailyDecisionMatrix[time, time] += 0-idx_value
                dailyDecisionMatrix[time+idx+1, time] += idx_value

        result.append(dailyDecisionMatrix)


    real_decision = pd.DataFrame(columns=('day', 'start', 'end', 'start_action', 'start_price', 'end_action', 'end_price'))
    for day in range(181):
        dt = result[day]
        # print dt.shape #368, 368: corresponding time, prediction time

        action_times = pd.DataFrame(columns=('day', 'start', 'end', 'start_action', 'start_price', 'end_action', 'end_price'))
        i = 0
        for t in range(31, 350, 1):
            if len(np.where(dt[:, t] != 0)[0]) > 0:
                (start_time, end_time) = np.sort(np.where(dt[:, t] != 0)[0])
                start_action = dt[:, t][start_time]
                end_action = dt[:, t][end_time]
                start_price = true_present[day*319+start_time-31]
                end_price = true_present[day*319+end_time-31]

                action_times.loc[i] = [day, start_time, end_time, start_action, start_price, end_action, end_price]
                i += 1

        k = 0
        valid_rows = [True] * action_times.shape[0]
        while k < action_times.shape[0]:
            end_value = action_times['end'].values[k]
            invalid_rows = action_times[action_times['start'] < end_value].index.tolist()
            for row in invalid_rows:
                if row > k:
                    valid_rows[row] = False

            k = max(invalid_rows) + 1

        real_decision_day = action_times.iloc[valid_rows].reset_index(drop=True)
        real_decision = pd.concat([real_decision, real_decision_day])

    real_decision['benefit'] = real_decision['start_action'] * real_decision['start_price'] + real_decision['end_action'] * real_decision['end_price'] - 0.03

    print sum(real_decision['benefit'])
    print sum(real_decision['benefit']) * 250000

    return real_decision

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    print threshold
    printBenefit(threshold, EXP_NO)