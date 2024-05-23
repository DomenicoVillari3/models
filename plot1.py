import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
df = pd.read_csv('SM4TMEAN.csv')

# Estrai le colonne di interesse
esecuzione = df['esecuzione']
avg_wer = df['avg_wer']
avg_time = df['avg_time']
avg_accuracy = df['avg_accuracy']

# Funzione per creare un istogramma
def create_bar_plot(x, y, title, ylabel, color):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x=x, height=y, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Esecuzione')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')  # Ruota le etichette dell'asse x
    plt.yscale('log')  # Utilizza una scala logaritmica sull'asse y
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Aggiungi i valori delle barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()  # Aggiusta automaticamente gli elementi del grafico
    plt.show()

# Crea l'istogramma per avg_wer
create_bar_plot(esecuzione, avg_wer, 'avg_wer', 'avg_wer', 'skyblue')

# Crea l'istogramma per avg_time
create_bar_plot(esecuzione, avg_time, 'avg_time', 'avg_time', 'salmon')

# Crea l'istogramma per avg_accuracy
create_bar_plot(esecuzione, avg_accuracy, 'avg_accuracy', 'avg_accuracy', 'green')
