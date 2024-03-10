import matplotlib.pyplot as plt

data = []
with open("eps_results.txt") as f:
    for line in f:
        data.append([float(x) for x in line.split()])

size = len(data)
x = [0 for i in range(size)]
t = [0 for i in range(size)]
e = [0 for i in range(size)]
for i in range(size):
    x[i], t[i], e[i] = data[i][0], data[i][1], data[i][2]

plt.title("Зависимость относительной погрешности от значения eps")
plt.xlabel("eps")
plt.ylabel("err")
plt.plot(x, e)
plt.show()
