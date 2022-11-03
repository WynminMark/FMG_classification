import PZcalibration as PZ
import pandas as pd

if __name__ == '__main__':
    file_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20221102"
    save_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20221102\final\data.xlsx"
    data_type = {'C':"C", 'P':"P"}
    sensor_num = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]
    test_num = ["-1", "-2", "-3"]

    final_data_writter = pd.ExcelWriter(save_path)
    for i in sensor_num:
        for j in test_num:
            #capacitance_file_path = file_path + '\\' + data_type['C'] + i + j + '.xls'
            #pressure_file_path = file_path + '\\' + data_type['P'] + i + j + '.txt'
            capacitance_file_path = file_path + '\\' + i + j + '.xls'
            pressure_file_path = file_path + '\\' + i + j + '.txt'
            try:
                temp_data = PZ.Z_P_calibration(capacitance_file_path, pressure_file_path, 1000)
                temp_data.to_excel(final_data_writter, sheet_name = i + j, index = True)
                print(capacitance_file_path)
            except FileNotFoundError:
                print("File not found err: ", capacitance_file_path)
                continue

    final_data_writter.save()
    final_data_writter.close()
    # print(file_path + '\\' + data_type[0] + sensor_num[0] + test_num[0])

    pass


# FileNotFoundError