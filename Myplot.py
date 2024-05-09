import matplotlib.pyplot as plt
import numpy as np 



def my_plot(wer,t):
    mean_wer = np.mean(wer)
    mean_time = np.mean(t)

    plt.ylabel('WER')
    plt.plot(wer, label='WER ', color='red',marker="o")
    plt.grid(True) 
    plt.legend()
    plt.axhline(y=mean_wer, color='green', linestyle='--', label=f'Valore Medio: {mean_wer:.2f}')  # Linea orizzontale per il valore medio
    plt.show()
    plt.savefig("fig/werfig.png")
    plt.clf()
    
    plt.ylabel('Time')
    plt.plot(t, label='Time' ,marker="o")
    plt.grid(True) 
    plt.legend()    
    plt.axhline(y=mean_time, color='green', linestyle='--', label=f'Valore Medio: {mean_time:.2f}')
    plt.show()
    plt.savefig("fig/timefig.png")
    plt.clf()