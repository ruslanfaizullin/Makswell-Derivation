import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageChops import offset
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import xlabel, ylabel
import scipy.integrate as spi
from matplotlib.ticker import NullLocator
from scipy.ndimage import label


R = 8.314
temp = input('Temperature (in Kelvins):\n')
molar = input('Molar mass (in kg/mol):\n')

#calculation of a recommendation for boarders
def func(x):
    return (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp)))**1.5) * np.exp(-(float(molar) * (x**2))/(2 * R * float(temp)))
x = 300

while True:
    value = func(x)
    if value < 0.00005:
        rec = x
        break
    x += 300


#calculation of the defined integral
boarders = list(map(int, input(f'Write boarders for the x-axis or length of x-axis(example, 0 1000). Recommendation is 0 {rec+200}:\n').split()))
intervals_results = []
for i in range(2, 11, 2):
    def integration(x, temp, molar):
        return (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp)))**1.5) * np.exp(-(float(molar) * (x**2))/(2 * R * float(temp)))

    low_limit = boarders[0] + (i-2)*((boarders[1]-boarders[0])//10)
    high_limit = i*((boarders[1]-boarders[0])//10)
    result, error = spi.quad(func = integration, a = low_limit, b = high_limit, args=(temp, molar))
    intervals_results.append(round(result, 2))

#check if sum of the probability more than one
if sum(intervals_results) > 1:
    delta = sum(intervals_results)-1
    maxN = max(intervals_results)
    ind = intervals_results.index(max(intervals_results))
    maxN-=delta
    intervals_results[ind] = round(maxN, 2)

#create the graph representation of the result
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()

x = np.arange(float(boarders[0]), float(boarders[1]), 2)
y = (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp))) ** 1.5) * np.exp(-(float(molar) * (x ** 2)) / (2 * R * float(temp)))
for j in range(2, 11, 2):
    low_limit = boarders[0] + (j - 2) * ((boarders[1] - boarders[0]) // 10)
    high_limit = j * ((boarders[1] - boarders[0]) // 10)
    mask = (x >= low_limit) & (x <= high_limit)
    color = ['b', 'r', 'g', 'm', 'y']
    plt.fill_between(x[mask], y[mask], color = f"{color[j//2 - 1]}")
    ax.plot(x[mask], y[mask], color=f"{color[j//2 - 1]}")

#creaing intervals of speeds which will be used as integration boarders
intervals = []
for k in range(0, 11, 2):
    intervals.append((boarders[1]//10) * k)

plt.figtext(0.65, 0.6, f"RESULTS\n\n {intervals_results[0]} of all molecules move with speeds\n in the range between {intervals[0]} m/s and {intervals[1]} m/s\n {intervals_results[1]} of all molecules move with speeds\n in the range between {intervals[1]} m/s and {intervals[2]} m/s\n {intervals_results[2]} of all molecules move with speeds\n in the range between {intervals[2]} m/s and {intervals[3]} m/s\n {intervals_results[3]} of all molecules move with speeds\n in the range between {intervals[3]} m/s and {intervals[4]} m/s\n {intervals_results[4]} of all molecules move with speeds\n in the range between {intervals[4]} m/s and {intervals[5]} m/s\n ", bbox = {'boxstyle': 'round', 'facecolor': 'white'})
ax.set_title("Makswell's distribution function for speeds (1 mol)")
ax.set_xlabel('speed in m/s')
ax.set_ylabel('probability density')
ax.grid()
plt.show()

#intervals_results - probabilities derivation for intervals of speeds, intervals - intervals of speeds
N = int(input('Write the integer number of molecules for visualization.Recommendation is 100 molecules\n'))
intervals_graph_speeds = [i/10000 for i in intervals]

xx = []
y = []
Vx = []
Vy = []

#create x, y, Vx and Vy for each interval molecules by using random module
for z in range(5):
    xn = np.random.normal(0, 3, int(N*intervals_results[z]))
    yn = np.random.normal(0, 3, int(N*intervals_results[z]))

    Vxn = []
    Vyn = []
    for _ in range(int(N*intervals_results[z])):
        Vxn.append(np.random.choice([np.random.uniform(intervals_graph_speeds[z], intervals_graph_speeds[z+1]), np.random.uniform(-(intervals_graph_speeds[z+1]), -(intervals_graph_speeds[z]))]))
        Vyn.append(np.random.choice([np.random.uniform(intervals_graph_speeds[z], intervals_graph_speeds[z+1]), np.random.uniform(-(intervals_graph_speeds[z+1]), -(intervals_graph_speeds[z]))]))
    Vx.append(Vxn)
    Vy.append(Vyn)
    xx.append(xn)
    y.append(yn)


Vx = [np.array(v) for v in Vx]
Vy = [np.array(v) for v in Vy]
xx = [np.array(xi) for xi in xx]
y = [np.array(yi) for yi in y]


# Boarders of the box with molecules
xmin, xmax = -10, 10
ymin, ymax = -10, 10

#create a graphs for different sorts of molecules
fig, ax = plt.subplots()
scat0 = ax.scatter(xx[0], y[0], color = 'b', label = f"{int(N*intervals_results[0])} molecules with speeds between {intervals[0]} m/s and {intervals[1]} m/s ")
scat1 = ax.scatter(xx[1], y[1], color = 'r', label = f"{int(N*intervals_results[1])} molecules with speeds between {intervals[1]} m/s and {intervals[2]} m/s ")
scat2 = ax.scatter(xx[2], y[2], color = 'g', label = f"{int(N*intervals_results[2])} molecules with speeds between {intervals[2]} m/s and {intervals[3]} m/s ")
scat3 = ax.scatter(xx[3], y[3], color = 'm', label = f"{int(N*intervals_results[3])} molecules with speeds between {intervals[3]} m/s and {intervals[4]} m/s ")
scat4 = ax.scatter(xx[4], y[4], color = 'y', label = f"{int(N*intervals_results[4])} molecules with speeds between {intervals[4]} m/s and {intervals[5]} m/s ")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(loc ='upper right')
ax.xaxis.set_major_locator(NullLocator())
ax.yaxis.set_major_locator(NullLocator())

def update(frame):
    global xx, y, Vx, Vy

    #Update the coordinates
    for c in range(5):
        xx[c] += Vx[c]
        y[c] += Vy[c]

    # Check when there should be a wall reflection
    # Check if coordinates not in the frames then change the sign of the speed
    for b in range(5):
        Vx[b][(xx[b] < xmin) | (xx[b] > xmax)] *= -1
        Vy[b][(y[b] < ymin) | (y[b] > ymax)] *= -1


    # Limit the coordinates to change the direction on the boarders
    for a in range(5):
        xx[a] = np.clip(xx[a], xmin, xmax)
        y[a] = np.clip(y[a], ymin, ymax)


    # Update points position
    offsets0 = np.column_stack((xx[0], y[0]))
    scat0.set_offsets(offsets0)

    offsets1 = np.column_stack((xx[1], y[1]))
    scat1.set_offsets(offsets1)

    offsets2 = np.column_stack((xx[2], y[2]))
    scat2.set_offsets(offsets2)

    offsets3 = np.column_stack((xx[3], y[3]))
    scat3.set_offsets(offsets3)

    offsets4 = np.column_stack((xx[4], y[4]))
    scat4.set_offsets(offsets4)

    return [scat0, scat1, scat2, scat3, scat4]

animation = FuncAnimation(
    fig,
    update,
    frames=200,
    interval=30,
    blit=False,
    repeat=True
)

plt.show()
