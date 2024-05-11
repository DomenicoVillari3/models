import matplotlib.pyplot as plt
import numpy as np 



def my_plot(wer,t,model):
    mean_wer = np.mean(wer)
    mean_time = np.mean(t)

    plt.ylabel('WER')
    plt.plot(wer, label='WER ', color='red',marker="o")
    plt.grid(True) 
    plt.legend()
    plt.axhline(y=mean_wer, color='green', linestyle='--', label=f'Valore Medio: {mean_wer:.2f}')  # Linea orizzontale per il valore medio
    plt.show()
    plt.savefig("fig/werfig{}.png".format(model))
    plt.clf()
    
    plt.ylabel('Time')
    plt.plot(t, label='Time' ,marker="o")
    plt.grid(True) 
    plt.legend()    
    plt.axhline(y=mean_time, color='green', linestyle='--', label=f'Valore Medio: {mean_time:.2f}')
    plt.show()
    plt.savefig("fig/timefig{}.png".format(model))
    plt.clf()

def plot_accuracy(model,accuracy):
    mean_accuracy = np.mean(accuracy)
    plt.ylabel('Accuracy')
    plt.plot(accuracy, label='Accuracy' ,marker="o",color='purple')
    plt.grid(True) 
    plt.legend()    
    plt.axhline(y=mean_accuracy, color='green', linestyle='--', label=f'Valore Medio: {mean_accuracy:.2f}')
    plt.show()
    plt.savefig("fig/accuracyfig{}.png".format(model))
    plt.clf()
