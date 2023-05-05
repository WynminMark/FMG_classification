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
import pylab

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
    * 根据时间戳计算每秒内FMG数据的平均值，用于与压力数据对应
    * 输入：
        db_file_path: FMG数据存储文件
        channel_num: 数据通道编号，ch0是时间，ch1-8对应压力通道
    * 输出：
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
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame({"time_stamp": i, "FMG": temp_mean_FMG}, index=[0])], ignore_index=True)
        pass
    return result_dataframe


def plate_func(P, r, t, h, D, C0, b):
    '''
    用于curve_fit方法拟合方程中的参数，但是无法计算协方差？
    C = 0.5*pi*C0*(r - (64*D*h*(1 + 0.488*h^2/t^2)/P)^0.25)^2
    r, t: substrate layer
    h: thickness of space layer
    D: bending stiffness
    C0: capacitance per unit area
    b: bias constant
    '''
    return C0*np.power((r - np.power(64*D*h*(1 + 0.488*np.power(h, 2)/np.power(t, 2))/P, 0.25)), 2) + b


def form_FMG_P(db_file_path, pressure_file_path, FMG_channel = 1, time_bias = 0):
    '''
    * 读取FMG和pressure数据，处理获得一次校准后的FMG-Pressure数据
    * db_file_path
    * pressure_file_path
    * FMG_channel
    * time_bias (second): FMG_time_stamp = pressure_time_stamp - time_bias
    * OUTPUT: Pressure, FMG array
    '''
    # 读取压力数据
    pressure = PZ.read_pressure(pressure_file_path)
    # 对每秒内的FMG数据取平均数，返回以秒为单位的时间戳和对应的FMG数据
    FMG_mean = mean_FMG_1s(db_file_path, FMG_channel) # dataframe

    # init result DataFrame
    final_data = pd.DataFrame()

    # 处理压力数据时间戳
    prss_timestamp_list = []

    for i in range(pressure.shape[0]):
        prss_timestamp_list.append(get_prss_timestamp(pressure['date'][i] + " " + pressure['time'][i]))
        pass

    # 对应FMG数值与气压值
    for i in range(pressure.shape[0]):
        for j in range(FMG_mean.shape[0]):
            if FMG_mean["time_stamp"][j] == prss_timestamp_list[i] - time_bias:
                final_data = pd.concat([final_data, pd.DataFrame({'time': pressure['time'][i],
                                                                'P/mmHg': pressure['pressure'][i],
                                                                'FMG': FMG_mean["FMG"][j]}, index=[0])], ignore_index=True)
                break

    FMG = final_data['FMG'].values
    P1 = final_data['P/mmHg'].values

    max_FMG_index = np.where(P1 == max(P1))[0][0]

    return P1[0 : max_FMG_index], FMG[0 : max_FMG_index]


class FMG2pressure():
    def __init__(self, FMG_channel = 1, time_bias = 0, **file_path):
        '''
        * 调用form_FMG_P()
        * file_path: [FMG.db, pressure.txt]...
        '''
        # 初始化列表用于存储几次校准数据
        original_pressure_list = []
        original_FMG_list = []
        for f_name in file_path.values():
            temp_p, temp_FMG = form_FMG_P(f_name[0], f_name[1], FMG_channel, time_bias)
            original_pressure_list.append(temp_p)
            original_FMG_list.append(temp_FMG)
            pass
        # 合并几次校准的原始数据
        self.original_P = np.hstack(original_pressure_list)
        self.original_FMG = np.hstack(original_FMG_list)
        # 存储文件路径
        self.file_path_dict = file_path
        pass

    def show_original(self):
        """
        * 显示校准散点图，删除异常数值
        """
        plt.figure()
        plt.scatter(self.original_FMG, self.original_P, label = "calibration output")
        plt.title("Pressure--FMG")
        plt.xlabel("FMG output")
        plt.ylabel("Pressure")
        plt.show()
        pass

    def get_fit_curve(self, deg = 3):
        '''
        * 获得多项式拟合方程，显示拟合曲线
        * deg: 方程最高次数
        * self.fit_equation(x)可以获得数据 x 对应的 pressure 值
        '''
        # 曲线拟合，返回值为多项式的各项系数
        fit_coefficients = np.polyfit(self.original_FMG, self.original_P, deg)
        # 返回值为多项式的表达式，也就是函数式子
        self.fit_equation = np.poly1d(fit_coefficients)
        # 根据函数的多项式表达式，求解 y
        y_pred = self.fit_equation(self.original_FMG)
        # print(np.polyval(p1, 29))             根据多项式求解特定 x 对应的 y 值
        # print(np.polyval(z1, 29))             根据多项式求解特定 x 对应的 y 值

        # 对拟合曲线的数据进行排序，避免画出多条曲线
        fit_curve_df = pd.DataFrame({'x': self.original_FMG, 'y': y_pred})
        sort_fit_curve_df = fit_curve_df.sort_values(by = "y")

        plot1 = pylab.plot(self.original_FMG, self.original_P, '*', label='original values')
        plot2 = pylab.plot(sort_fit_curve_df['x'], sort_fit_curve_df['y'], 'r', label='fit values')
        pylab.title('')
        pylab.xlabel('')
        pylab.ylabel('')
        pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0.65, 0.05))
        pylab.show()
        pass

    def get_pressure_unit(self):
        pressure = PZ.read_pressure(list(self.file_path_dict.values())[0][1])   # 取第一个[.db, .txt]中的.txt
        return pressure['unit'].values[0]

    # end class
    pass


if __name__ == '__main__':
    db_file_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20230326传感器校准数据\3-1.db"
    pressure_file_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20230326传感器校准数据\3-1.txt"

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
                final_data = pd.concat([final_data, pd.DataFrame({'time': pressure['time'][i],
                                                                'P/mmHg': pressure['pressure'][i],
                                                                'FMG': FMG_mean["FMG"][j]}, index=[0])], ignore_index=True)
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
    print(P1[0 : max_FMG_index], FMG[0 : max_FMG_index])

