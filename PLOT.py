import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel

R = 8.314

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()

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

boarders = list(map(int, input(f'Write boarders for the x-axis(example, 0 1000). Recommendation is 0 {rec+200}:\n').split()))

x = np.arange(float(boarders[0]), float(boarders[1]), 2)
y = (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp))) ** 1.5) * np.exp(-(float(molar) * (x ** 2)) / (2 * R * float(temp)))

ax.plot(x, y, color='m')
ax.set_title("Makswell's distribution function for speeds (1 mol)")
ax.set_xlabel('speed in m/s')
ax.set_ylabel('probability density')
ax.grid()
plt.show()
