import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def my_plot(csv_file,model):
     # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract data
    wer = df['wer'].values
    t = df['time'].values
    accuracy = df['accuracy'].values


    mean_wer = np.mean(wer)
    mean_time = np.mean(t)
    mean_accuracy = np.mean(accuracy)
    
    # Plotting WER
    plt.figure()
    plt.plot(wer, label='WER', color='red', marker="o")
    plt.axhline(y=mean_wer, color='green', linestyle='--', label=f'Mean: {mean_wer:.2f}')
    plt.xlabel('Index')
    plt.ylabel('WER')
    plt.grid(True)
    plt.legend()
    plt.title('{} WER Plot'.format(model))
    plt.savefig(f"fig/werfig{model}.png")
    plt.show()
    plt.clf()
    
    # Plotting Time
    plt.figure()
    plt.plot(t, label='Time', color='blue', marker="o")
    plt.axhline(y=mean_time, color='green', linestyle='--', label=f'Mean: {mean_time:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.grid(True)
    plt.legend()
    plt.title('{} Time Plot'.format(model))
    plt.savefig(f"fig/timefig{model}.png")
    plt.show()
    plt.clf()

    # Plotting Accuracy
    plt.figure()
    plt.plot(accuracy, label='Accuracy', color='purple', marker="o")
    plt.axhline(y=mean_accuracy, color='green', linestyle='--', label=f'Mean: {mean_accuracy:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.title('{} Accuracy Plot'.format(model))
    plt.savefig(f"fig/accuracyfig{model}.png")
    plt.show()
    plt.clf()
