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


num = 1  # change
all_data = [get_data(i) for i in range(num)]
all_x = [all_data[i][0] for i in range(num)]
all_y = [all_data[i][1] for i in range(num)]
for i in range(num):
    plt.scatter(all_x[i], all_y[i], s=2)

plt.xlabel('Значение eps')  # change
plt.ylabel('Ошибка')  # change
plt.plot(all_x[0], all_y[0], linewidth=1, label='sqrt(x) + 2 * y')
# plt.plot(all_x[1], all_y[1], linewidth=1, label='1 / sin(x + y + 0.1)')
# plt.plot(all_x[2], all_y[2], linewidth=1, label='1500')
# plt.plot(all_x[3], all_y[3], linewidth=1, label='128')
# plt.plot(all_x[4], all_y[4], linewidth=1, label='256')

plt.title("Зависимость ошибки от значения eps\nфункции sqrt(x) + 2 * y при N=200")  # change

# plt.legend()
plt.show()
