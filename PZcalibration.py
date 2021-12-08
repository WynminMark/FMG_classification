# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:06:51 2021

@author: WeimyMark

用于处理气压计软件保存的 .txt 气压数据和 LCR meter保存的 excel, 并生成自动校准的数据
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def read_pressure(file_path):
    # input: handheld气压计软件保存的txt数据
    # air pressure data in dataframe
    with open(file_path) as f:
        data = f.readlines()
    
    newData = []
    for lines in data:
        a = lines.split()
        newData.append(a)
    
    newData.pop(0)#删除第一行txt列名
    P_data = pd.DataFrame(newData,
                          columns = ["index", "date", "time", "pressure", "unit"],
                          dtype = float)
    return P_data

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


def Z_P_calibration(Z_path, P_path):
    # 处理气压计txt数据和LCRmeter xlsx数据
    # 按照时间对应自动对齐，返回dataframe
    impedance = pd.read_excel(Z_path)
    pressure = read_pressure(P_path)
    final_data = pd.DataFrame()
    
    for i in range(0, pressure.shape[0]):
        for j in range(0, impedance.shape[0]):
            #若时间一致，挑出数据并储存
            if pressure['time'][i] == str(impedance.loc[j][5]).split()[1]:
                final_data = final_data.append({'time': pressure['time'][i],
                                                'P/mmHg': pressure['pressure'][i],
                                                'Cs/nF': impedance.loc[j][2]*1e12,
                                                'Rs/kΩ': impedance.loc[j][3]*1e-3,
                                                'Z': (impedance.loc[j][3]**2 + 1/(2*math.pi*1000*impedance.loc[j][2]*1e3)**2)**0.5},
                                               ignore_index = True)
    return final_data


def cal_z(R, w, C):
    # 并联电阻R和C计算阻抗绝对值
    z = ((R/(1 + R**2 * w**2 * C**2))**2 + (R**2 * w * C/(1 + R**2 * w**2 * C**2))**2)**0.5
    return z


if __name__ == "__main__":
    # 读FMG数据文件
    raw_data = pd.read_table("D:\code\data\Z_P\s3-3.db", sep = ';', header = None)
    FMG = raw_data[6].values# 获得有效的一路压力信号
    rsmp_FMG, k = zip_signal(FMG, 1230)# 降采样到1Hz，与压力采样率对应
    # 读压力数据
    pressure = read_pressure("D:\code\data\Z_P\s3-3.txt")
    P = pressure['pressure'].values
    # 读LCR数据system output vs LCR meter
    LCR_data = Z_P_calibration("D:\code\data\Z_P\s3-6.xlsx", "D:\code\data\Z_P\s3-6.txt")
    P2 = LCR_data['P/mmHg'].values
    Z = LCR_data['Z'].values
    max_index = np.where(P2 == max(P2))
    # 用LCR meter测量值计算
    cal_FMG = []
    for i in Z[0:30]:
        j = 865048800/i     #310280136
        cal_FMG.append(j)
    
    
    # 以下根据实际信号情况调整
    plt.figure()
    plt.plot(rsmp_FMG)
    plt.title("FMG signal")
    plt.show()
    
    plt.figure()
    plt.plot(P)
    plt.title("air pressure(mmHg)")
    plt.show()
    
    max_P_index = np.where(P == max(P))
    max_FMG_index = np.where(rsmp_FMG == max(rsmp_FMG))
    print("P, FMG: ", max_P_index, max_FMG_index)
    
    plt.figure()
    plt.plot(P[0:42], rsmp_FMG[0:42])
    plt.title("system calibration")
    plt.xlabel("pressure (mmHg)")
    plt.ylabel("ADC values")
    plt.show()
    
    
    plt.figure()
    plt.plot(P2[0:30], cal_FMG, label = "Z-->system output")
    plt.plot(P[0:42], rsmp_FMG[0:42], label = "system output")
    plt.legend(["Z-->system output", "system output"])
    plt.title("Z-P")
    plt.xlabel("pressure (mmHg)")
    plt.ylabel("Z")
    plt.show()
    
    # print(cal_z(330000, 2*math.pi*1000, 47e-12))
    
    
    
    