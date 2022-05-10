import csv

import pulp
from pulp import *
from pulp import GLPK
import numpy as np
import matplotlib.pyplot as plt

PARK_MAXIMUM_POWER = 7000
BATTERY_EFFICIENCY = 0.85
n_charge = 0.93
BATTERY_POWER_CAPACITY = 0.45 * PARK_MAXIMUM_POWER / n_charge
n_discharge = 0.89
forecasting_st_dev = [0.07, 0.15, 0.25]
shortage_penalty = 0.6
excess_discount = 0.4


def ceil_div(a, b):
    return -(a // -b)


def get_num_days():
    filename = 'ForecastingData.csv'
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=';')
    smp = []
    forecasted_production = []
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        smp.append(float(row[4]))
        forecasted_production.append(float(row[1]))
        line_count += 1
    return ceil_div(len(smp), 24)


def random_solve():
    filename = 'ForecastingData.csv'

    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter = ';')
    smp = []
    forecasted_production = []
    actual_production = [[] for i in range(len(forecasting_st_dev))]

    line_count = 0

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        smp.append(float(row[4]))
        forecasted_production.append(float(row[1]))
        line_count += 1

    forecasting_error = [[] for i in range(len(forecasting_st_dev))]
    for z in range(len(forecasting_st_dev)):
        for i in range(len(forecasted_production)):
            actual_production[z].append(np.random.normal(forecasted_production[i], abs(forecasted_production[i])
                                                          * forecasting_st_dev[z]))
            if actual_production[z][i] > PARK_MAXIMUM_POWER:
                actual_production[z][i] = PARK_MAXIMUM_POWER
            forecasting_error[z].append(forecasted_production[i] - actual_production[z][i])


    '''
    # Plot forecasted vs actual production
    hour = [i+1 for i in range(len(smp))]
    plt.plot(hour, forecasting_error, 'rs', label="forecasted production")
    plt.plot(hour,actual_production,'bo',label="actual production")
    plt.legend(loc="upper right")
    plt.xlabel("days")
    plt.show()
    '''

    e_c_stored = [[]for i in range(len(forecasting_st_dev))]
    e_d_stored = [[]for i in range(len(forecasting_st_dev))]
    soc_stored = [[]for i in range(len(forecasting_st_dev))]
    objective_yearly = [[]for i in range(len(forecasting_st_dev))]
    objective_daily2 = [[] for i in range(len(forecasting_st_dev))]
    # global num_days
    num_days = ceil_div(len(smp), 24)
    for z in range(len(forecasting_st_dev)):
        for i in range(num_days):
            e_c = []
            e_d = []
            battery_state_lp_var = []
            model = LpProblem("forecasting_error_model", sense=LpMaximize)
            for k in range(24):
                e_c.append(LpVariable(name='e_c_' + str(k), lowBound=0, cat=const.LpInteger))
                e_d.append(LpVariable(name='e_d_' + str(k), lowBound=0, cat=const.LpInteger))
                battery_state_lp_var.append(LpVariable(
                    name='battery_state_lp_var' + str(k), lowBound=0, upBound=BATTERY_POWER_CAPACITY, cat=const.LpInteger))

                if forecasting_error[z][24*i+k] > 0:
                    model += e_d[k] == 0
                    model += e_c[k] <= forecasting_error[z][24 * i + k]
                elif forecasting_error[z][24*i+k] < 0:
                    model += e_c[k] == 0
                    model += e_d[k] <= abs(forecasting_error[z][24 * i + k])
                '''
                else:
                    model += e_d[k] == 0
                    model += e_c[k] == 0 '''

                if k == 0:
                    model += (battery_state_lp_var[0] == BATTERY_POWER_CAPACITY / 2)
                else:
                    model += (battery_state_lp_var[k] == battery_state_lp_var[k - 1] + e_c[k - 1] - e_d[k - 1])
                if k == 23:
                    model += (battery_state_lp_var[k] == BATTERY_POWER_CAPACITY / 2)

            model += ((shortage_penalty * lpDot(e_d, smp[24*i:24*(i+1)]) - excess_discount * lpDot(e_c, smp[24*i:24*(i+1)])
                      )/1000)
            status = model.solve(PULP_CBC_CMD(msg=False))
            for k in range(24):
                e_d_stored[z].append(e_d[k].value())
                e_c_stored[z].append(e_c[k].value())
                soc_stored[z].append(battery_state_lp_var[k].value())
            objective_daily2[z].append(model.objective.value())
        objective_yearly[z] = (shortage_penalty * lpDot(e_d_stored[z], smp) - excess_discount * lpDot(e_c_stored[z], smp)) / 1000
        #print(len(objective_yearly[z]))
        #print(len(objective_daily2[z]))
        #print("objective value 1 year:", objective_yearly[z])
        #print("objective value 1 year2:", lpSum(objective_daily2[z]))

    return objective_daily2


# Plot daily
def plot_daily(daily_values1, daily_values2, daily_values3):
    num_days = [i for i in range(len(daily_values1))]
    plt.plot(num_days, daily_values1, 'r', label="daily comparison " + str(forecasting_st_dev[0]))
    plt.plot(num_days, daily_values2, 'b', label="daily comparison " + str(forecasting_st_dev[1]) )
    plt.plot(num_days, daily_values3, 'g', label="daily comparison " + str(forecasting_st_dev[2]))
    plt.legend(loc="upper left")
    plt.xlabel("days")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()


def convert_to_month_values(v):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    current_month = 0

    v_by_month = []
    i = 0
    while i < len(v):
        current_month_days = month_days[current_month]
        v_by_month.append(sum(v[i:i + current_month_days]))
        i += current_month_days
        current_month += 1

    return v_by_month


def plot_monthly(daily_values1, daily_values2, daily_values3):
    month_values1 = convert_to_month_values(daily_values1)
    month_values2 = convert_to_month_values(daily_values2)
    month_values3 = convert_to_month_values(daily_values3)
    months = [i for i in range(12)]
    plt.plot(months, month_values1, 'r', label="monthly comparison " + str(forecasting_st_dev[0]))
    plt.plot(months, month_values2, 'b', label="monthly comparison " + str(forecasting_st_dev[1]))
    plt.plot(months, month_values3, 'g', label="monthly comparison " + str(forecasting_st_dev[2]))
    plt.legend(loc="upper left")
    plt.xlabel("months")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()


if __name__ == '__main__':

    num_days = get_num_days()
    num_simulations = 5
    objective_daily = [ [ [] for j in range(num_days) ] for i in range(len(forecasting_st_dev)) ]

    for z in range(num_simulations):
        solved = random_solve()
        for i in range(len(forecasting_st_dev)):
            for j in range(num_days):
                objective_daily[i][j].append(solved[i][j])
    # print(objective_daily)

    mean_objective_daily = [[ 0 for j in range(num_days) ] for i in range(len(forecasting_st_dev))]
    for i in range(len(forecasting_st_dev)):
        for j in range(num_days):
            mean_objective_daily[i][j] = np.mean(objective_daily[i][j])
    print("mean " + str(mean_objective_daily))
    print("lpSum " + str(lpSum(mean_objective_daily[0])))
    print("lpSum " + str(lpSum(mean_objective_daily[1])))
    print("lpSum " + str(lpSum(mean_objective_daily[2])))

    # print(pulp.listSolvers(onlyAvailable=True))
    plot_daily(mean_objective_daily[0], mean_objective_daily[1], mean_objective_daily[2])
    plot_monthly(mean_objective_daily[0], mean_objective_daily[1], mean_objective_daily[2])

    print("bye")