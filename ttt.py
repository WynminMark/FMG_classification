import pandas as pd
import matplotlib.pyplot as plt
import math

def read_pressure(file_path):
    # input: handheld气压计软件保存的txt数据文件路径
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

def cal_z(R, w, C):
    # 并联电阻R和C计算阻抗绝对值
    # w == 2*pi*f
    z = ((R/(1 + R**2 * w**2 * C**2))**2 + (R**2 * w * C/(1 + R**2 * w**2 * C**2))**2)**0.5
    return z
'''
impedance = pd.read_excel(r"D:\LEARNINNNNNNNNNNNNG\实验数据\20220421\123.xls")
pressure = read_pressure(r"D:\LEARNINNNNNNNNNNNNG\实验数据\20220421\p7-1.txt")
final_data = pd.DataFrame()

print(pressure['time'])
print('******************************')
print(str(impedance.loc[0][5]))
'''
if __name__ == '__main__':
    '''
    R = 432000
    C = 0.7e-9
    Z = (R**2 + 1/(2*math.pi*1000*C)**2)**0.5
    print(Z)
    '''
    print(cal_z(330000, 2*math.pi*600, 47e-12))
