import csv
from pulp import *
import numpy as np
import matplotlib.pyplot as plt

PARK_MAXIMUM_POWER = 24000

n_charge = 0.93
n_discharge = 0.93
BATTERY_ENERGY_CAPACITY = 0.25 * PARK_MAXIMUM_POWER / n_charge
BATTERY_TO_GENERATION_CAPACITY = 0.25 * PARK_MAXIMUM_POWER
BATTERY_EFFICIENCY = n_charge * n_discharge
smp_forecasting_uncertainty = [30, 50, 80]
iterations_num = 300
epsilon = 0.5

def ceil_div(a, b):
    return -(a // -b)


if __name__ == '__main__':

    # Read data from csv

    filename = 'TestData2.csv'
    # filename = 'C:\Users\fokio\Documents\...'
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=';')

    smp = []
    production = []
    line_count = 0
    for row in csv_reader:
        if line_count <= 0:
            line_count += 1
            continue

        smp.append(float(row[2]))
        production.append(float(row[3]))
        line_count += 1

    # Setup daily optimization problem

    e_w_stored = []
    e_c_stored = []
    e_d_stored = []
    soc_stored = []
    objective_daily2 = []
    objective_comparison2 = []
    num_days = ceil_div(len(smp), 24)
    num_weeks = ceil_div(len(smp), 7*24)
    day = [i + 1 for i in range(num_days)]
    week = [i + 1 for i in range(num_weeks)]
    temp = BATTERY_ENERGY_CAPACITY
    for i in range(num_days):
        e_w = []
        e_c = []
        e_d = []
        battery_state_lp_var = []
        model = LpProblem("on_storage_model", sense=LpMaximize)
        for k in range(24):
            e_w.append(LpVariable(name='e_w_' + str(k), lowBound=0, cat=const.LpInteger))
            e_c.append(LpVariable(name='e_c_' + str(k), lowBound=0, upBound=BATTERY_TO_GENERATION_CAPACITY, cat=const.LpInteger))
            e_d.append(LpVariable(name='e_d_' + str(k), lowBound=0, upBound=BATTERY_TO_GENERATION_CAPACITY, cat=const.LpInteger))
            battery_state_lp_var.append(LpVariable(
                name='battery_state_lp_var' + str(k), lowBound=0, upBound=BATTERY_ENERGY_CAPACITY, cat=const.LpInteger))
            model += (e_w[k] + e_c[k] <=  production[24 * i + k] + epsilon)
            model += production[24 * i + k] - epsilon <= e_w[k] + e_c[k]
            model += (e_w[k] + e_d[k] <= 1.00 * PARK_MAXIMUM_POWER)
            if k == 0:
                model += (battery_state_lp_var[0] == temp)
            else:
                model += (battery_state_lp_var[k] == battery_state_lp_var[k - 1] + e_c[k-1] - e_d[k-1])
            '''
            if k == 23:
                model += (battery_state_lp_var[k] == BATTERY_ENERGY_CAPACITY / 2)
                model += e_d[k] == 0
            '''
        # Objective function

        model += (lpDot(smp[24*i:24*(i+1)], e_w) + BATTERY_EFFICIENCY * lpDot(smp[24*i:24*(i+1)], e_d)) / 1000
        status = model.solve(PULP_CBC_CMD(msg=False))
        # Store daily values of decision variables and objective function in lists

        for k in range(24):
            e_d_stored.append(e_d[k].value())
            e_c_stored.append(e_c[k].value())
            e_w_stored.append(e_w[k].value())
            soc_stored.append(battery_state_lp_var[k].value())
        objective_daily2.append(model.objective.value())
        temp = battery_state_lp_var[23].value()
        objective_comparison2.append((lpDot(smp[24*i:24*(i+1)], production[24*i:24*(i+1)]))/1000)

    objective_yearly = (lpDot(smp, e_w_stored) + BATTERY_EFFICIENCY * lpDot(smp, e_d_stored)) / 1000
    objective_comparison = lpDot(smp, production) / 1000
    ''''
    print(len(objective_yearly))
    print(len(objective_comparison))
    print("objective value 1 year:", objective_yearly)
    print("this number should be the same:", lpSum(objective_daily2))
    print("objective comparison 1 year:", objective_comparison)
    print("this number should be the same:", lpSum(objective_comparison2))
    '''

    # Deviate SMP prices due to 24h Forecasting Error

    deviated_smp = np.zeros((len(smp_forecasting_uncertainty), iterations_num, len(smp)))
    for z in range(len(smp_forecasting_uncertainty)):
        for k in range(iterations_num):
            for i in range(len(smp)):
                deviated_smp[z][k][i] = (np.random.triangular(smp[i] - smp_forecasting_uncertainty[z], smp[i], abs(smp[i]) + smp_forecasting_uncertainty[z]))
                '''
                if deviated_smp[z][k][i] < 0:
                    deviated_smp[z][k][i] = 0
                '''


    deviated_objective = np.zeros((len(smp_forecasting_uncertainty), iterations_num))

    # Write results in CSV

    csv_output_file = open("TestDataOutput.csv", mode='w', newline='')
    csv_output_writer = csv.writer(csv_output_file, delimiter=';')
    for i in range(num_days):
        csv_output_writer.writerow([None, None, None, None, objective_daily2[i], objective_comparison2[i]])
    for i in range(len(smp)):
        csv_output_writer.writerow([e_w_stored[i], e_c_stored[i], e_d_stored[i], soc_stored[i],
                                    None, None, smp[i], deviated_smp[2][0][i]])

    # Yearly deviated objective value

    max_deviated_objective = np.zeros(len(smp_forecasting_uncertainty))
    min_deviated_objective = np.zeros(len(smp_forecasting_uncertainty))
    for z in range(len(smp_forecasting_uncertainty)):
        for k in range(iterations_num):
            deviated_objective[z][k] = (np.dot(deviated_smp[z][k]/1000, e_w_stored) + BATTERY_EFFICIENCY
                                        * np.dot(deviated_smp[z][k]/1000, e_d_stored))

    for z in range(len(smp_forecasting_uncertainty)):
        max_deviated_objective[z] = np.max(deviated_objective[z])
        min_deviated_objective[z] = np.min(deviated_objective[z])
    # print(deviated_objective)
    # print(max_deviated_objective)
    # Convert to daily values
    i = 0
    l = 0
    weekly_comparison = []
    weekly_objective = []
    weekly_deviated_objective = np.zeros((len(smp_forecasting_uncertainty), num_weeks, iterations_num))
    mean_weekly_deviated_objective = np.zeros((len(smp_forecasting_uncertainty), num_weeks))
    while i < len(smp):
        weekly_comparison.append((lpDot(smp[i:i + 24*7], production[i:i + 24*7]).value()) / np.sum(production[i:i + 24*7]))
        weekly_objective.append((
            lpDot(smp[i:i+24*7], e_w_stored[i:i+24*7]).value()
            + BATTERY_EFFICIENCY * lpDot(smp[i:i+24*7], e_d_stored[i:i+24*7]).value()) / np.sum(production[i:i + 24*7])
                                )
        for z in range(len(smp_forecasting_uncertainty)):
            for k in range(iterations_num):
                weekly_deviated_objective[z][l][k] = ((
                                                    np.dot(deviated_smp[z][k][i:i + 24*7], e_w_stored[i:i + 24*7])
                                                    + BATTERY_EFFICIENCY * np.dot(deviated_smp[z][k][i:i + 24*7], e_d_stored[
                                                    i:i + 24*7])) / np.sum(production[i:i + 24*7])
                                                      )
            mean_weekly_deviated_objective[z][l] = np.mean(weekly_deviated_objective[z][l])
        i += 24 * 7
        l += 1

    # month values
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    current_month = 0

    l = 0
    monthly_comparison = []
    monthly_objective = []
    monthly_feed_in_tariff = []
    fit_rate = np.zeros(len(smp))
    for i in range(len(smp)):
        fit_rate[i] = 55
    monthly_deviated_objective = np.zeros((len(smp_forecasting_uncertainty), len(month_days), iterations_num))
    max_monthly_deviated_objective = np.zeros((len(smp_forecasting_uncertainty), len(month_days)))
    min_monthly_deviated_objective = np.zeros((len(smp_forecasting_uncertainty), len(month_days)))
    i = 0
    while i < len(smp):
        month_hours = month_days[current_month] * 24

        monthly_comparison.append((np.dot(smp[i:i+month_hours], production[i:i+month_hours]))/PARK_MAXIMUM_POWER)
        monthly_feed_in_tariff.append((np.dot(fit_rate[i:i+month_hours], production[i:i+month_hours]))/PARK_MAXIMUM_POWER)
        monthly_objective.append((
            np.dot(smp[i:i+month_hours], e_w_stored[i:i+month_hours])
            + BATTERY_EFFICIENCY * np.dot(smp[i:i+month_hours], e_d_stored[i:i+month_hours]))/PARK_MAXIMUM_POWER)
        for z in range(len(smp_forecasting_uncertainty)):
            for k in range(iterations_num):
                monthly_deviated_objective[z][l][k] = ((
                                                      np.dot(deviated_smp[z][k][i:i + month_hours],
                                                            e_w_stored[i:i + month_hours])
                                                      + BATTERY_EFFICIENCY * np.dot(deviated_smp[z][k][i:i + month_hours],
                                                         e_d_stored[i:i + month_hours]))/PARK_MAXIMUM_POWER)

            max_monthly_deviated_objective[z][l] = np.max(monthly_deviated_objective[z][l])
            min_monthly_deviated_objective[z][l] = np.min(monthly_deviated_objective[z][l])
        i += month_hours
        l += 1
        current_month += 1

    #print(weekly_comparison)

    # Plot daily

    plt.plot(week, weekly_comparison, 'r--', label="weekly comparison")
    plt.plot(week, weekly_objective, 'b+', label="weekly objective")
    plt.plot(week, mean_weekly_deviated_objective[0], 'go', label="weekly deviated objective 15%")
    plt.plot(week, mean_weekly_deviated_objective[1], 'yo', label="weekly deviated objective 20%")
    # plt.plot(week, mean_weekly_deviated_objective[2], 'orange', label="weekly deviated objective 35%")
    plt.legend(loc="upper left")
    plt.xlabel("weeks")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()
    print(len(monthly_comparison))
    # Plot monthly
    # num_months = ceil_div(len(smp), 744)
    num_months = 12
    month = [i + 1 for i in range(num_months)]
    plt.plot(month, monthly_comparison, 'r--', label="monthly comparison")
    plt.plot(month, monthly_objective, 'b+', label="monthly objective")
    plt.plot(month, monthly_feed_in_tariff, 'g', label="monthly FIT")
    # plt.plot(month, min_monthly_deviated_objective[1], 'orange', label="monthly deviated objective 20%")
    # plt.plot(month, max_monthly_deviated_objective[2], 'bo', label="monthly deviated objective 35%")
    plt.legend(loc="upper left")
    plt.xlabel("months")
    plt.ylabel("Revenue [Eur/MW]")
    plt.show()
    soc_mean = np.mean(soc_stored)
        #results
    print("soc mean:", soc_mean/BATTERY_ENERGY_CAPACITY)
    print(f"hybrid energy price:", objective_yearly/np.sum(production)*1000)
    print("wind energy price:", objective_comparison/np.sum(production)*1000)
    print(f"hybrid revenue:", objective_yearly)
    print("wind revenue:", objective_comparison)
    print("extra revenue: ", objective_yearly-objective_comparison)
    print("deviated objective 1:", np.mean(deviated_objective[0]))
    print("deviated objective 2:", np.mean(deviated_objective[1]))
    print("deviated objective 3:", np.mean(deviated_objective[2]))

    csv_output_file = open("ArbitrageGraphs2.csv", mode='w', newline='')
    csv_output_writer = csv.writer(csv_output_file, delimiter=';')
    header = ['month', 'smp dev 10% max', 'smp dev 20% max', 'smp dev 10% min', 'smp dev 20% min', 'objective', 'comparison','feed in tariff']
    csv_output_writer.writerow(header)
    for i in range(num_months):
        csv_output_writer.writerow([month[i], max_monthly_deviated_objective[0][i], max_monthly_deviated_objective[1][i]
                                       , min_monthly_deviated_objective[0][i], min_monthly_deviated_objective[1][i],
                                    monthly_objective[i], monthly_comparison[i],monthly_feed_in_tariff[i]])



