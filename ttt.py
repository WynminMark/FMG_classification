import pandas as pd
import matplotlib.pyplot as plt

raw_FMG = pd.read_table(r"d:\code\data\iFMG_calibration\t1.db",  sep = ';', header = None)

FMG = raw_FMG[1].values

plt.figure()
plt.plot(FMG)
plt.show()