import matplotlib.pyplot as plt
import numpy as np
import statistics

# names of plots base on number of generated sites
n = "10"
name_of_file = "output_" + n +".txt"
name_avg = "plot_aver_" + n +".png"
name_greedy = "plot_greedy_" + n +".png"
name_dsatur = "plot_dsatur_" + n +".png"
name_rlf = "plot_rlf_" + n +".png"

res = []
with open(name_of_file, "r") as file:
    for line in file:
        res.append(list(map(int, line.split())))

# calculate the standard deviation for each group
transposed = list(zip(*res))
std_devs = [statistics.stdev(group) for group in transposed]


# init values
num_greedy = {i: 0 for i in range(1, 8)}
num_dsatur = {i: 0 for i in range(1, 8)}
num_rlf = {i: 0 for i in range(1, 8)}
aver_greedy, aver_dsatur, aver_rlf = 0, 0, 0

# set values
for r in res:
    aver_greedy += r[0]
    aver_dsatur += r[1]
    aver_rlf += r[2]
    num_greedy[r[0]] += 1
    num_dsatur[r[1]] += 1
    num_rlf[r[2]] += 1

# --------------------------------

# plot: average number of colors per algorithm
len_10 = len(res)
x = np.array(["Greedy", "DSatur", "RLF"])
y = np.array([aver_greedy/len_10, aver_dsatur/len_10, aver_rlf/len_10])
bar_colors = ['tab:green', 'tab:blue', 'tab:orange']

plt.xlabel("Algorytm")
plt.ylabel("Srednia liczba użytych kolorów")

plt.bar(x,y,color=bar_colors)
plt.errorbar(x,y, yerr=std_devs, fmt="o", color="r")
plt.savefig(name_avg, format="png", dpi=300, bbox_inches="tight")
plt.clf()

# --------------------------------

# function for adding number of occurrences above each bar
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i+1, y[i], y[i], ha = 'center')

# plot: number of colors using Greedy algorithm
val_1 = [val for val in num_greedy.values()]
val_1_total = sum(val_1)
val_1_perc = [round(val/val_1_total*100, 0) for val in val_1]
colors = [i + 1 for i in range(7)]

plt.bar(colors, val_1, label='Greedy')
plt.title('Greedy')
plt.xlabel('Liczba kolorów')
plt.ylabel('Liczba wystąpień')
addlabels(colors, val_1)
plt.savefig(name_greedy, format="png", dpi=300, bbox_inches="tight")
plt.clf()


# plot: number of colors using DSatur algorithm
val_2 = [val for val in num_dsatur.values()]
val_2_total = sum(val_2)
val_2_perc = [round(val/val_2_total*100, 0) for val in val_2]
colors = [i + 1 for i in range(7)]

plt.bar(colors, val_2, label='DSatur')
plt.title('DSatur')
plt.xlabel('Liczba kolorów')
plt.ylabel('Liczba wystąpień')
addlabels(colors, val_2)
plt.savefig(name_dsatur, format="png", dpi=300, bbox_inches="tight")
plt.clf()


# plot: number of colors using RLF algorithm
val_3 = [val for val in num_rlf.values()]
val_3_total = sum(val_3)
val_3_perc = [round(val/val_3_total*100, 0) for val in val_3]
colors = [i + 1 for i in range(7)]

plt.bar(colors, val_3, label='RLF')
plt.title('RLF')
plt.xlabel('Liczba kolorów')
plt.ylabel('Liczba wystąpień')
addlabels(colors, val_3)
plt.savefig(name_rlf, format="png", dpi=300, bbox_inches="tight")