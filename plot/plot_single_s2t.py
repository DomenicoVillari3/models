import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def my_plot(csv_file,model):
    df=pd.read_csv(csv_file)
    entry=df["entry"]
    wer=df["wer"]
    time=df["time"]
    accuracy=df["accuracy"]
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',"tab:purple"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(entry,wer)

    plt.title(model+" Wer")
    plt.xlabel("Esecuzione")
    plt.ylabel("WER")
    plt.xticks(rotation=45, ha='right', fontsize=0)
    plt.grid()
    plt.tight_layout()
    plt.show()

my_plot("/home/domenico/Scrivania/Models/models/whisper-large-v3.csv","whisper-large-v3.csv")