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
import time

def read_pressure(file_path):
    # input: handheld气压计软件保存的txt数据
    # output: air pressure data in dataframe
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


def get_prss_timestamp(time_str):
    timeArray = time.strptime(time_str, "%m-%d-%Y %H:%M:%S")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp


def get_FMG_timestamp(time_str):
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp

def FMG_P_calibration(FMG_path, pressure_path, FMG_channel):
    raw_FMG = pd.read_table(FMG_path, sep = ';', header = None)
    pressure = read_pressure(pressure_path)
    
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
    return final_data


if __name__ == "__main__":
    # 获得FMG-P数据
    FMG_P_data = FMG_P_calibration(r"d:\code\data\iFMG_calibration\t3.db", r"d:\code\data\iFMG_calibration\t3.txt", 1)
    FMG = FMG_P_data['FMG'].values
    P1 = FMG_P_data['P/mmHg'].values
    # 读LCR数据system output vs LCR meter
    LCR_data = Z_P_calibration(r"d:\code\data\iFMG_calibration\t6.xls", r"d:\code\data\iFMG_calibration\t6.txt")
    P2 = LCR_data['P/mmHg'].values
    Z = LCR_data['Z'].values
    R= LCR_data['Rs/kΩ'].values

    # 以下根据实际信号情况调整

    max_FMG_index = np.where(P1 == max(P1))[0][0]
    max_LCR_index = np.where(P2 == max(P2))[0][0]
    
    # 用LCR meter测量值计算
    # 1KHz反馈阻抗328444
    # 直流反馈阻抗330000
    # 0.92665076 是传输前计算的比值0.9266U=output
    cal_FMG = []
    Rf = cal_z(330000, 2*math.pi*1000, 47e-12)
    gen2 = 2**0.5
    cons_c = 1000*4095/(1248*3300)
    for i in range(max_LCR_index):
        j = (328444000/Z[i]) * 0.92665076    #310280136
        #j = (Rf*200/(gen2*Z[i]) + 1000)*cons_c
        cal_FMG.append(j)

    plt.figure()
    plt.plot(P2[0 : max_LCR_index], cal_FMG, label = "Z-->system output")
    plt.plot(P1[0 : max_FMG_index], FMG[0 : max_FMG_index], label = "system output")
    plt.legend(["Z-->system output", "system output"])
    plt.title("Z-P")
    plt.xlabel("pressure (mmHg)")
    plt.ylabel("Z")
    plt.show()
    
    print("P1: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
    for i in P1[0 : max_FMG_index]:
        print(i)

    print("FMG: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
    for i in FMG[0 : max_FMG_index]:
        print(i)
        
    print("P2: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
    for i in P2[0 : max_LCR_index]:
        print(i)
        
    print("LCR: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" )
    for i in cal_FMG:
        print(i)
    