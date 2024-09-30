import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_bars(x,y,xlab,ylab,title):
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',"tab:purple"]
    # Plotting WER
    plt.figure(figsize=(10, 6))

    bars = plt.bar(x,y,color=bar_colors, edgecolor="black",label=x)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(x)
    plt.xticks(rotation=45, ha='right', fontsize=0)
    plt.grid()
    plt.tight_layout()
    plt.show()

def s2t_comparison(filename):
    df=pd.read_csv(filename)
    wer=df["avg_wer"].values
    model_names=df["modello"].values
    time=df["avg_time"].values
    accuracy=df["avg_accuracy"].values

    create_bars(model_names,wer,"Modello","WER","WER Comparison")

    create_bars(model_names,time,"Modello","Tempo (s)","Time Comparison")

    create_bars(model_names,accuracy,"Modello","Accuracy", "Accuracy Comparison")
    
def translation_comparison(filename):
    df=pd.read_csv(filename)
    bleu=df["bleu_score"].values
    model_names=df["model"].values
    time=df["avg_time"].values
    
    create_bars(model_names,bleu,"Modello","BLEU Score","BLEU Score Comparison")

    create_bars(model_names,time,"Modello","Tempo (s)","Time Comparison")





s2t_comparison('/home/domenico/Scrivania/Models/models/tesi/csv/meansM1.csv')
#translation_comparison('/home/domenico/Scrivania/Models/models/translate_means.csv')