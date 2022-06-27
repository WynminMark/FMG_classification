import PZcalibration as PZ
import pandas as pd

if __name__ == '__main__':
    file_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20220621"
    save_path = r"D:\LEARNINNNNNNNNNNNNG\ExperimentData\20220621\fianl\data.xlsx"
    data_type = {'C':"C", 'P':"P"}
    sensor_num = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010","011", "012", "013", "014", "015"]
    test_num = ["-1", "-2", "-3"]

    final_data_writter = pd.ExcelWriter(save_path)
    for i in sensor_num:
        for j in test_num:
            capacitance_file_path = file_path + '\\' + data_type['C'] + i + j + '.xls'
            pressure_file_path = file_path + '\\' + data_type['P'] + i + j + '.txt'
            print(capacitance_file_path)
            temp_data = PZ.Z_P_calibration(capacitance_file_path, pressure_file_path, 1000)
            temp_data.to_excel(final_data_writter, sheet_name = i + j, index = True)
    final_data_writter.save()
    final_data_writter.close()
    # print(file_path + '\\' + data_type[0] + sensor_num[0] + test_num[0])

    pass
