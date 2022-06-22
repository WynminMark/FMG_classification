import pandas as pd
import matplotlib.pyplot as plt

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

impedance = pd.read_excel(r"D:\LEARNINNNNNNNNNNNNG\实验数据\20220421\123.xls")
pressure = read_pressure(r"D:\LEARNINNNNNNNNNNNNG\实验数据\20220421\p7-1.txt")
final_data = pd.DataFrame()

print(pressure['time'])
print('******************************')
print(str(impedance.loc[0][5]))

