import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy import stats


def convert_int(i):
    mid = (i[0] + i[1]) / 2
    t = (i[1] - i[0]) / 2
    if t >= 1e-3:
        return f'${mid:.3f}\pm {t:.3f}$'
    else:
        return f'${mid:.3f}$'


def get_data(j):
    with open(f"results_init/results_sin_{j}.txt") as f:  # change
        n, runs = map(int, f.readline().split())
        x, y, ints, errs = [], [], [], []
        for i in range(n):
            size = int(f.readline())
            x.append(size)
            t = list(map(float, f.readline().split()))
            mean = np.mean(t)
            std = np.std(t, ddof=1)
            interval = stats.t.interval(confidence=0.95, df=len(t) - 1, loc=mean, scale=stats.sem(t))
            y.append(mean)
            ints.append(interval)
            errs.append(std / mean)
    return x, y, ints, errs


num = 3  # change
mp = {0: 'нули', 1: 'среднее значение', 2: 'случайные значения'}  # change
all_data = [get_data(i) for i in range(num)]

# graph
all_x = [all_data[i][0] for i in range(num)]
all_y = [all_data[i][1] for i in range(num)]

plt.xlabel('Размер сетки')  # change
plt.ylabel('Время, сек')  # change
plt.title("Сравнение производительности при разных размерах"
          "\nсетки для функции sin(x)+sin(y)")  # change

for i in range(num):
    plt.plot(all_x[i], all_y[i], linewidth=1, label=mp[i])
    plt.scatter(all_x[i], all_y[i], s=2)
plt.legend()
plt.show()

# table with intervals
all_int = [all_data[i][2] for i in range(num)]
table = PrettyTable(junction_char='|')
table.field_names = [''] + [str(n) for n in all_x[0]]

for i in range(num):
    table.add_row([mp[i]] + [convert_int(inter) for inter in all_int[i]])

f = open('results_init/table_sin.txt', 'w')  # change
print(table, file=f)

# check errs
all_errs = [all_data[i][3] for i in range(num)]
mx = 0
for i in range(num):
    for it in all_errs[i]:
        mx = max(mx, it)
print(f'Максимум по отношениям стандартного отклонения к среднему в %: {mx * 100}', file=f)
