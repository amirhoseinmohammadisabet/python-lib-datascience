import math
import numpy as np


def integration(integrand, lower, upper, *args):
    panels = 10
    limits = [lower, upper]
    h = (limits[1] - limits[0]) / (2 * panels)
    n = (2 * panels) + 1
    x = np.linspace(limits[0], limits[1], n)
    y = integrand(x, *args)
    # Simpson 1/3
    I = 0
    start = -2
    for looper in range(0, panels):
        start += 2
        counter = 0
        for looper in range(start, start + 3):
            counter += 1
            if counter == 1 or counter == 3:
                I += ((h / 3) * y[looper])
            else:
                I += ((h / 3) * 4 * y[looper])
    return I


def f(x, a, b):
    return a * pow(x, b)


I = integration(f, 1, math.pi / 2, 1, 2)
print(I)
