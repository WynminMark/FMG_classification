import pandas as pd
import os
'''
author: @ZhouKeji
'''
#文件读取

def Open_file(directory):

    file_array = [] #用于储存所有文件
    file_num = 0    #用于记录文件数量
    
    #读取路径内所有文件
    for file in os.listdir(directory):
        file_num = file_num + 1
        file = directory + '\\' + file
        file_array.append(file)
    
    #根据文件命名规律：奇数为txt文件，偶数为xls和xlsx文件
    f_txt = [] #用于储存txt
    f_xls = [] #用于储存xls和xlsx
    for i in range(int(file_num/2)): #循环次数是文件总数的一半
        
        #打开txt文件
        with open(file_array[2*i]) as f:
            data1 = f.readlines()
        newData1 = []
        for lines in data1:
            a = lines.split()
            newData1.append(a)
        newData1 = pd.DataFrame(newData1)
        f_txt.append(newData1)
        
        #打开excel文件
        data2 = pd.read_excel(file_array[2*i+1])
        f_xls.append(data2)
        
    return f_txt,f_xls,int(file_num/2)


#路径设置
Path_open = r'D:\呼吸阻塞定位\数据\20210923\9-2'
Path_save = r'D:\呼吸阻塞定位\数据\20210923\9-2'

#打开文件
Data_txt,Data_xls,num = Open_file(Path_open)

#循环处理每一对文件
for k in range(num):
    
    #每次循环处理一次
    data_txt = Data_txt[k]
    data_xls = Data_xls[k]
    
    #创建空的DataFrame
    FinalData = pd.DataFrame()
    
    #两个大循环，一一匹对
    for i in range(1,len(data_txt)):
        for j in range(1,len(data_xls)):
            #若时间一致，挑出数据并储存
            if data_txt[2][i] == str(data_xls.loc[j][5]).split()[1]:
                FinalData = FinalData.append({'time':data_txt[2][i],
                                              'P/mmHg':data_txt[3][i],
                                              'Cs/nF':data_xls.loc[j][2]*1e12,
                                              'Rs/Ω':data_xls.loc[j][3]},
                                             ignore_index=True)
    
    #导出为excel
    FinalData.to_excel(Path_save+'\\FinalData'+str(k+1)+'.xls')
