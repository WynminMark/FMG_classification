# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:11:01 2021

@author: WeimyMark
"""

import PZcalibration as pz
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


def zip_signal(signal, fs):
    # input: FMG signal with a sample frequency of fs
    # ouput: FMG signal with a sample frequency of 1
    sig_len = len(signal)
    win_len = round(fs*0.1/2)*2    # 平滑滤波窗宽
    ave_sig = []    # 储存平滑后的信号
    for i in range(win_len//2, sig_len - win_len//2):
        ave_sig.append(sum(signal[i-win_len//2 : i+win_len//2])/win_len)
    
    max_index = np.where(ave_sig == max(ave_sig))[0][0] # 最大值的索引
    start_index = max_index % fs
    
    rsmp_sig = []# 储存降采样后的信号
    for i in range(start_index, max_index+1, fs):
        rsmp_sig.append(signal[i])
    return rsmp_sig, np.where(ave_sig == max(ave_sig))

raw_data = pd.read_table("D:\code\data\Z_P\s3-3.db", sep = ';', header = None)
FMG = raw_data[6].values# 获得有效的一路压力信号

rsmp_FMG, k = zip_signal(FMG, 1230)


plt.figure()
plt.plot(rsmp_FMG)
plt.title("ramp_FMG")
plt.show()
























