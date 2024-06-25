import pandas as pd
import matplotlib.pyplot as plt

# Carica il nuovo file CSV
df_bleu = pd.read_csv('marianTranlaste.csv')  # Cambia il nome del file con quello corretto se necessario

# Estrai le colonne di interesse
avg_bleu_mean = df_bleu['bleu_score'].mean()

# Calcola la deviazione standard per la colonna di interesse bleu_score
std_bleu = df_bleu['bleu_score'].std()

# Funzione per creare un istogramma con deviazione standard
def create_bar_plot_with_std(x, y_mean, y_std, title, ylabel, color, ylim=None):
    plt.figure(figsize=(6, 6))
    bars = plt.bar(x, y_mean, color=color, edgecolor='black', yerr=y_std, capsize=5)
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
    
    # Aggiungi la legenda
    legend_colors = [plt.Rectangle((0,0),1,1, color=color[i]) for i in range(len(color))]
    plt.legend(legend_colors, ['Media'], loc='upper left')
    
    plt.tight_layout()  # Aggiusta automaticamente gli elementi del grafico
    plt.show()

# Creiamo una lista di esecuzioni che includa solo il valore medio
esecuzione_with_std = ['Media']
avg_bleu_with_std = [avg_bleu_mean]
std_bleu_with_std = [std_bleu]

# Colori leggermente diversi per le barre
colors = ['skyblue']

# Crea l'istogramma per bleu_score con deviazione standard
create_bar_plot_with_std(esecuzione_with_std, avg_bleu_with_std, std_bleu_with_std, 'BLEU Score', 'bleu_score', colors,ylim=100)

