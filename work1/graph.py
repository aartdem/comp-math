import matplotlib.pyplot as plt


def get_data(j):
    data = []
    with open(f"results{j}.txt") as f:
        for line in f:
            data.append([float(x) for x in line.split()])
    size = len(data)
    x = [data[i][0] for i in range(size)]
    y = [data[i][1] for i in range(size)]
    return x, y


num = 3 # change
all_data = [get_data(i) for i in range(num)]
all_x = [all_data[i][0] for i in range(num)]
all_y = [all_data[i][1] for i in range(num)]
for i in range(num):
    plt.scatter(all_x[i], all_y[i], s=2)

plt.xlabel('Размер сетки (N)') # change
plt.ylabel('Время, сек') # change
plt.plot(all_x[0], all_y[0], linewidth=1, label='нули')
plt.plot(all_x[1], all_y[1], linewidth=1, label='среднее значение')
plt.plot(all_x[2], all_y[2], linewidth=1, label='случайные значения')

plt.title("Сравнение производительности при разных\nначальных приближениях "
          "для функции 1/sin(x+y+0.1), THREADS_NUM=4")
plt.legend()

plt.show()
