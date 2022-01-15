# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:11:01 2021

@author: WeimyMark
"""
import PZcalibration as PZ
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
data = pz.Z_P_calibration("D:\code\data\Z_P\s3-6.xlsx", "D:\code\data\Z_P\s3-6.txt")

P = data['P/mmHg'].values
Z = data['Z'].values
max_index = np.where(P == max(P))

plt.figure()
plt.plot(P[5:30], Z[5:30])
plt.title("Z-P")
plt.xlabel("pressure (mmHg)")
plt.ylabel("Z")
plt.show()
'''

def get_prss_timestamp(time_str):
    timeArray = time.strptime(time_str, "%m-%d-%Y %H:%M:%S")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp


def get_FMG_timestamp(time_str):
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp


raw_FMG = pd.read_table(r"d:\code\data\iFMG_calibration\t1.db",  sep = ';', header = None)
FMG_channel = 1
pressure = PZ.read_pressure(r"d:\code\data\iFMG_calibration\t1.txt")

final_data = pd.DataFrame()
'''
pressure_val = pressure['pressure'].values
temp_search_index = 0
max_search_index = np.where(pressure_val == max(pressure_val))[0][0] + 5
'''
FMG_timestamp_list = []
prss_timestamp_list = []

for i in range(pressure.shape[0]):
    prss_timestamp_list.append(get_prss_timestamp(pressure['date'][i] + " " + pressure['time'][i]))

for i in range(raw_FMG.shape[0]):
    FMG_timestamp_list.append(get_FMG_timestamp(raw_FMG[0][i]))

for i in range(pressure.shape[0]):
    for j in range(raw_FMG.shape[0]):
        if FMG_timestamp_list[j] == prss_timestamp_list[i] - 1:
            final_data = final_data.append({'time': pressure['time'][i],
                                            'P/mmHg': pressure['pressure'][i],
                                            'FMG': raw_FMG[FMG_channel][j]},
                                            ignore_index = True)
            break

            



FMG = final_data['FMG'].values
P1 = final_data['P/mmHg'].values

max_FMG_index = np.where(P1 == max(P1))[0][0]

plt.figure()
plt.plot(P1[0 : max_FMG_index], FMG[0 : max_FMG_index], label = "system output")
plt.title("Z-P")
plt.xlabel("pressure (mmHg)")
plt.ylabel("Z")
plt.show()


















