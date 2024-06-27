import pandas as pd
import numpy as np

# Leggi il file CSV
df = pd.read_csv('seamless-m4t-v2-large.csv')

wer_list = df["wer"]
time_list=df["time"]
acc_list=df["accuracy"]

from Myplot import my_plot
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

WriteMeanToCSV("means.csv","seamless-m4t-v2-large",avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 