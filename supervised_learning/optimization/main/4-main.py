#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
        data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64, 81, 71, 69,
                65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67, 67, 68, 75]
        days = list(range(1, len(data) + 1))
        m_avg = moving_average(data, 0.9)
        print(m_avg)
        plt.plot(days, data, 'r', days, m_avg, 'b')
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (Fahrenheit)')
        plt.title('SF Maximum Temperatures in October 2018')
        plt.legend(['actual', 'moving_average'])
        plt.show()

# Expected Output:
# [72.0, 75.15789473684211, 73.62361623616238,
#  71.98836871183484, 70.52604332006544, 70.20035470453027,
#  71.88706986789997, 73.13597603396988, 71.80782582850702,
#  70.60905915023126, 69.93737009120935, 71.0609712312634,
#  70.11422355031073, 69.32143707981284, 70.79208718739721,
#  70.81760741911772, 70.59946700377961, 69.9406328280786,
#  70.17873340222755, 69.47534437750306, 68.41139351151023,
#  67.58929643210207, 66.97601174673004, 66.86995043877324,
#  67.42263231561797, 67.91198666959514, 67.8151574064495,
#  67.72913996327617, 67.65262186609462, 67.68889744321645,
#  68.44900744806469]