#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, color='red')  # Plot y as a solid red line
plt.xticks(np.arange(11))  # Set x-axis ticks to match the range 0 to 10
plt.show()