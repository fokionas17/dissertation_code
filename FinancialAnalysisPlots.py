import csv
import matplotlib.pyplot as plt
import numpy as np


line_count = 0
datacolumns = 3
iterations = 80
if __name__ == '__main__':
    filename = 'PlotData.csv'
    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=';')
    data = np.zeros((datacolumns, iterations+1))
    actual_frequency = []
    for row in csv_reader:
        if line_count <= 0:
            line_count += 1
            continue

        for z in range(datacolumns):
            data[z][line_count-1] = float(row[z])
        actual_frequency.append(row[3])
        line_count += 1

plt.plot(data[0], actual_frequency, 'r', label=str(row[0][0]))
plt.plot(data[1], actual_frequency, 'b', label=str(row[0][1]))
plt.plot(data[2], actual_frequency, 'g', label=str(row[0][2]))
values = [0, 0.2, 0.4, 0.6, 0.8, 1]
#plt.yticks([])
#plt.xlim(-1,2)
#plt.xticks(actual_frequency, values)
plt.show()