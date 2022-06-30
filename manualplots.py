# SMP plots
import csv

import numpy as np

if __name__ == '__main__':

    # Read data from csv

    filename = 'TestData2.csv'
    # filename = 'C:\Users\fokio\Documents\...'
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=';')
    production = []
    line_count = 0
    for row in csv_reader:
        if line_count <= 0:
            line_count += 1
            continue
        production.append(float(row[3]))
        line_count += 1
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_production = np.zeros(len(month_days))
    denominator = np.zeros(len(month_days))
    days_passed = 0
    cf_per_month = np.zeros(len(month_days))
    for i in range(len(month_days)):
        monthly_production[i] = np.sum(production[24*days_passed:24*(days_passed+month_days[i])])
        denominator[i] = 24 * month_days[i] * 24000
        cf_per_month[i] = monthly_production[i]/denominator[i] * 100
        days_passed += month_days[i]
    cf_average = np.mean(cf_per_month)
    print(monthly_production)
    print(denominator)
    print(cf_per_month)
    print(cf_average)