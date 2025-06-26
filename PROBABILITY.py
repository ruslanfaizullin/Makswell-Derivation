import matplotlib.pyplot as plt
import scipy.integrate as spi
import numpy as np
R = 8.314

#calculating of the defined integral
temp = input('Temperature (in Kelvins):\n')
molar = input('Molar mass (in kg/mol):\n')
low_limit = float(input("Write smaller integration boarder:\n"))
high_limit = float(input("Write higher integration boarder:\n"))
def integration(x, temp, molar):
    return (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp)))**1.5) * np.exp(-(float(molar) * (x**2))/(2 * R * float(temp)))

result, error = spi.quad(func = integration, a = low_limit, b = high_limit, args=(temp, molar))

#creating the graph representation of the result
R = 8.314

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()

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

boarders = list(map(int, input(f'Write boarders for the x-axis or length of x-axis(example, 0 1000). Recommendation is 0 {rec+200} :\n').split()))

x = np.arange(float(boarders[0]), float(boarders[1]), 2)
y = (4 * np.pi * (x ** 2)) * ((float(molar) / (2 * np.pi * R * float(temp))) ** 1.5) * np.exp(-(float(molar) * (x ** 2)) / (2 * R * float(temp)))

mask = (x >= low_limit) & (x <= high_limit)

plt.fill_between(x[mask], y[mask], color = 'b')
plt.figtext(0.5, 0.6, f"RESULTS\n\n {round(result, 2)} of all molecules move with speeds\n in the range between {low_limit} m/s and {high_limit} m/s", bbox = {'boxstyle': 'round', 'facecolor': 'white'})
ax.set_title("Makswell's distribution function for speeds (1 mol)")
ax.set_xlabel('speed in m/s')
ax.set_ylabel('probability density')
ax.grid()
ax.plot(x, y, color='m')
plt.show()


