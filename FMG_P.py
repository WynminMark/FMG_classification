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
由FMG穿戴式系统保存的.db文件和气压计文件，获得FMG-pressure数据
'''

def get_prss_timestamp(time_str):
    """
    气压计软件handheld保存数据中的时间转换成时间戳，
    精度为秒
    """
    timeArray = time.strptime(time_str, "%m-%d-%Y %H:%M:%S")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp


def get_FMG_timestamp(time_str):
    """
    FMG输出.db文件中的time_str转时间戳，
    精度为秒
    """
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
    time_stamp = int(time.mktime(timeArray))
    return time_stamp


def mean_FMG_1s(db_file_path, channel_num):
    """
    根据时间戳计算每秒内FMG数据的平均值，用于与压力数据对应
    输入：
        db_file_path: FMG数据存储文件
        channel_num: 信号通道编号
    输出：
        时间戳和FMG数据的dataframe
    """
    raw_FMG = pd.read_table(db_file_path,  sep = ';', header = None)
    # 转换时间戳，精度为秒
    FMG_timestamp_list = []
    result_dataframe = pd.DataFrame(columns=("time_stamp", "FMG"))

    for i in range(raw_FMG.shape[0]):
        FMG_timestamp_list.append(get_FMG_timestamp(raw_FMG[0][i]))
        pass
        
    # 使用bool索引获得同一秒内的所有FMG数据
    time_stamp_array = np.array(FMG_timestamp_list)
    min_stamp = FMG_timestamp_list[0]
    max_stamp = FMG_timestamp_list[-1]
    for i in range(min_stamp, max_stamp+1, 1):
        temp_data = raw_FMG[channel_num].values[time_stamp_array == i]
        temp_mean_FMG = sum(temp_data)/len(temp_data)
        result_dataframe = result_dataframe.append({"time_stamp": i,
                                                    "FMG": temp_mean_FMG}, ignore_index = True)
        pass
    return result_dataframe


if __name__ == '__main__':
    db_file_path = r"d:\code\data\iFMG_calibration\t1.db"
    pressure_file_path = r"d:\code\data\iFMG_calibration\t1.txt"

    # 读取FMG数据和压力数据
    raw_FMG = pd.read_table(db_file_path,  sep = ';', header = None)
    FMG_channel = 1
    pressure = PZ.read_pressure(pressure_file_path)

    final_data = pd.DataFrame()

    FMG_timestamp_list = []
    prss_timestamp_list = []

    for i in range(pressure.shape[0]):
        prss_timestamp_list.append(get_prss_timestamp(pressure['date'][i] + " " + pressure['time'][i]))
        pass

    for i in range(raw_FMG.shape[0]):
        FMG_timestamp_list.append(get_FMG_timestamp(raw_FMG[0][i]))
        pass

    FMG_mean = mean_FMG_1s(db_file_path, 1) # dataframe

    # 对应FMG数值与气压值
    for i in range(pressure.shape[0]):
        for j in range(FMG_mean.shape[0]):
            if FMG_mean["time_stamp"][j] == prss_timestamp_list[i] - 1:
                final_data = final_data.append({'time': pressure['time'][i],
                                                'P/mmHg': pressure['pressure'][i],
                                                'FMG': FMG_mean["FMG"][j]}, ignore_index = True)
                break

    FMG = final_data['FMG'].values
    P1 = final_data['P/mmHg'].values

    max_FMG_index = np.where(P1 == max(P1))[0][0]

    plt.figure()
    plt.plot(P1[0 : max_FMG_index], FMG[0 : max_FMG_index], label = "system output")
    plt.title("OUTPUT-P")
    plt.xlabel("pressure (mmHg)")
    plt.ylabel("Output")
    plt.show()

