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

def FMG_P(FMG_path, pressure_path):
    raw_FMG = pd.read_table(FMG_path, sep = ';', header = None)
    pressure = PZ.read_pressure(pressure_path)
    
    final_data = pd.DataFrame()
    for i in range(pressure.shape[0]):
        for j in range(raw_FMG.shape[0]):
            if get_FMG_timestamp(raw_FMG[0][j]) == get_prss_timestamp(pressure['date'][i] + " " + pressure['time'][i]):
                final_data = final_data.append({'time': pressure['time'][i],
                                                'P/mmHg': pressure['pressure'][i],
                                                'FMG': raw_FMG[6][j]},
                                               ignore_index = True)
                break
    return final_data
            





















