import csv
from pulp import *
from pulp import GLPK
import numpy as np
import matplotlib.pyplot as plt

PARK_MAXIMUM_POWER = 7000
BATTERY_EFFICIENCY = 0.85
n_charge = 0.93
BATTERY_POWER_CAPACITY = 0.5 * PARK_MAXIMUM_POWER / n_charge
n_discharge = 0.89
smp_forecasting_st_dev = [0.15, 0.25, 0.35]
epsilon = 0.5

def ceil_div(a, b):
    return -(a // -b)


if __name__ == '__main__':

    #
    # Read data from csv
    #


    filename = 'TestData2.csv'
    # filename = 'C:\Users\fokio\Documents\...'

    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=';')

    smp = []
    production = []
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue

        smp.append(float(row[2]))
        production.append(float(row[3]))
        line_count += 1




    #
    # Setup optimization problem
    #
    e_w_stored = []
    e_c_stored = []
    e_d_stored = []
    soc_stored = []
    e_d_real = []
    battery_state = []
    objective_daily2 = []
    objective_comparison2 = []
    num_days = ceil_div(len(smp), 24)
    day = [i + 1 for i in range(num_days)]
    print(len(production))
    print(num_days)


    for i in range(num_days):
        e_w = []
        e_c = []
        e_d = []
        battery_state_lp_var = []
        model = LpProblem("on_storage_model", sense=LpMaximize)
        for k in range(24):
            e_w.append(LpVariable(name='e_w_' + str(k), lowBound=0, cat=const.LpInteger))
            e_c.append(LpVariable(name='e_c_' + str(k), lowBound=0, cat=const.LpInteger))
            e_d.append(LpVariable(name='e_d_' + str(k), lowBound=0, cat=const.LpInteger))
            battery_state_lp_var.append(LpVariable(
                name='battery_state_lp_var' + str(k), lowBound=0, upBound=BATTERY_POWER_CAPACITY, cat=const.LpInteger))
            model += (e_w[k] + e_c[k] == production[24 * i + k])
            model += (e_w[k] + e_d[k] <= PARK_MAXIMUM_POWER)
            if k == 0:
                model += (battery_state_lp_var[0] == 0)
            else:
                model += (battery_state_lp_var[k] == battery_state_lp_var[k - 1] + e_c[k-1] - e_d[k-1])
            if k == 23:
                model += (battery_state_lp_var[k] == 0)
                model += e_d[k] == 0
                #model += (battery_state_lp_var[k-1] + e_c[k] - e_d[k] == BATTERY_POWER_CAPACITY)
                #battery_state_lp_var[k] == battery_state_lp_var[k-1] + e_c[k] - e_d[k]
                #model += e_d[k] == 0
                #model += e_c[k] == 0

            # model += (e_w[k] + e_c[k] >= production[24 * i + k] - epsilon)

        '''
            if k < 2:
                continue
            else:
                model += (e_d[k] + e_d[k - 1] + e_d[k - 2] - e_c[k - 2] >= 0)'''

        model += (lpDot(smp[24*i:24*(i+1)], e_w) + BATTERY_EFFICIENCY * lpDot(smp[24*i:24*(i+1)], e_d)) / 1000

        status = model.solve(PULP_CBC_CMD(msg=False))
        for k in range(24):
            e_d_stored.append(e_d[k].value())
            e_c_stored.append(e_c[k].value())
            e_w_stored.append(e_w[k].value())
            soc_stored.append(battery_state_lp_var[k].value())
        objective_daily2.append(model.objective.value())
        objective_comparison2.append((lpDot(smp[24*i:24*(i+1)], production[24*i:24*(i+1)]))/1000)


    objective_yearly = (lpDot(smp, e_w_stored) + BATTERY_EFFICIENCY * lpDot(smp, e_d_stored)) / 1000
    objective_comparison = lpDot(smp, production) / 1000
    print(len(objective_yearly))
    print(len(objective_comparison))
    print("objective value 1 year:", objective_yearly)
    print(lpSum(objective_daily2))
    print("objective comparison 1 year:", objective_comparison)
    print(lpSum(objective_comparison2))
    csv_output_file = open("TestDataOutput.csv", mode='w', newline='')
    csv_output_writer = csv.writer(csv_output_file, delimiter=';')
    for i in range(num_days):
        csv_output_writer.writerow([None, None, None, None, objective_daily2[i], objective_comparison2[i]])

    deviated_smp = [[] for i in range(len(smp_forecasting_st_dev))]
    for z in range(len(smp_forecasting_st_dev)):
        for i in range(len(smp)):
            deviated_smp[z].append(np.random.normal(smp[i], abs(smp[i]) * smp_forecasting_st_dev[z]))
            if deviated_smp[z][i] < 0:
                deviated_smp[z][i] = 0

    deviated_objective = [[] for i in range(len(smp_forecasting_st_dev))]
    for i in range(len(smp)):
        csv_output_writer.writerow([e_w_stored[i], e_c_stored[i], e_d_stored[i], soc_stored[i],
                                    None, None, smp[i], deviated_smp[0][i]])
    for z in range(len(smp_forecasting_st_dev)):
        deviated_objective[z] = (lpDot(deviated_smp[z], e_w_stored) + BATTERY_EFFICIENCY * lpDot(deviated_smp[z], e_d_stored))/1000


    # Convert to daily values
    i = 0
    daily_comparison = []
    daily_objective = []
    daily_deviated_objective = [[] for i in range(len(smp_forecasting_st_dev))]
    while i < len(smp):
        daily_comparison.append((lpDot(smp[i:i+24], production[i:i+24]).value())/PARK_MAXIMUM_POWER)
        daily_objective.append((
            lpDot(smp[i:i+24], e_w_stored[i:i+24]).value()
            + BATTERY_EFFICIENCY * lpDot(smp[i:i+24], e_d_stored[i:i+24]).value())/PARK_MAXIMUM_POWER
        )
        for z in range(len(smp_forecasting_st_dev)):
            daily_deviated_objective[z].append((
                                                lpDot(deviated_smp[z][i:i + 24], e_w_stored[i:i + 24]).value()
                                                + BATTERY_EFFICIENCY * lpDot(deviated_smp[z][i:i + 24], e_d_stored[
                                                i:i + 24]).value()) / PARK_MAXIMUM_POWER
                                        )
        i += 24

    # month values
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    current_month = 0
    i = 0
    monthly_comparison = []
    monthly_objective = []
    monthly_deviated_objective = [[] for i in range(len(smp_forecasting_st_dev))]
    while i < len(smp):
        month_hours = month_days[current_month] * 24

        monthly_comparison.append((lpDot(smp[i:i+month_hours], production[i:i+month_hours]).value())/PARK_MAXIMUM_POWER)
        monthly_objective.append((
            lpDot(smp[i:i+month_hours], e_w_stored[i:i+month_hours]).value()
            + BATTERY_EFFICIENCY * lpDot(smp[i:i+month_hours], e_d_stored[i:i+month_hours]).value())/PARK_MAXIMUM_POWER
        )
        for z in range(len(smp_forecasting_st_dev)):
            monthly_deviated_objective[z].append((
                                                  lpDot(deviated_smp[z][i:i + month_hours],
                                                        e_w_stored[i:i + month_hours]).value()
                                                  + BATTERY_EFFICIENCY * lpDot(deviated_smp[z][i:i + month_hours],
                                                                               e_d_stored[
                                                                               i:i + month_hours]).value()) / PARK_MAXIMUM_POWER
                                          )

        i += month_hours
        current_month += 1

    #print(daily_comparison)

    # Plot daily

    plt.plot(day, daily_comparison, 'r', label="daily comparison")
    plt.plot(day, daily_objective, 'b', label="daily objective")
    plt.plot(day, daily_deviated_objective[0], 'g', label="daily deviated objective 30%")
    plt.legend(loc="upper left")
    plt.xlabel("days")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()

    # Plot monthly
    # num_months = ceil_div(len(smp), 744)
    num_months = 12
    month = [i + 1 for i in range(num_months)]
    plt.plot(month, monthly_comparison, 'r--', label="monthly comparison")
    plt.plot(month, monthly_objective, 'b+', label="monthly objective")
    plt.plot(month, monthly_deviated_objective[0], 'go', label="monthly deviated objective 30%")
    plt.legend(loc="upper left")
    plt.xlabel("months")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()

        #results
    print(f"objective:", objective_yearly)
    print("objective comparison:", objective_comparison)
    print("deviated objective 1:", deviated_objective[0])

