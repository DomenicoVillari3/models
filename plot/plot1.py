import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
df = pd.read_csv('sm4tTranslate.csv')

# Estrai le colonne di interesse
esecuzione = df['esecuzione']
avg_wer = df['bleu_score']

#avg_accuracy = df['avg_accuracy']

# Funzione per creare un istogramma
def create_bar_plot(x, y, title, ylabel, color,ylim=None):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x=x, height=y, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Esecuzione')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')  # Ruota le etichette dell'asse x
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if ylim is not None:
        plt.ylim(0, ylim)  # Imposta il limite dell'asse
    
    # Aggiungi i valori delle barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()  # Aggiusta automaticamente gli elementi del grafico
    plt.show()

# Crea l'istogramma per avg_wer
create_bar_plot(esecuzione, avg_wer, 'bleu_score', 'bleu_score', 'skyblue',ylim=100)

# Crea l'istogramma per avg_time
#create_bar_plot(esecuzione, avg_time, 'avg_time', 'avg_time', 'salmon',ylim=5)

# Crea l'istogramma per avg_accuracy
#create_bar_plot(esecuzione, avg_accuracy, 'avg_accuracy', 'avg_accuracy', 'green')
