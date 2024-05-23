import pandas as pd
import matplotlib.pyplot as plt

# Carica i file CSV
df_whisper = pd.read_csv('whisper.csv')
df_whispercpp = pd.read_csv('whispercpp.csv')
df_sm4t = pd.read_csv('SM4T.csv')

# Calcola le medie per ciascun modello
means = {
    'model': ['Whisper', 'WhisperCPP', 'SM4T'],
    'WER': [df_whisper['wer'].mean(), df_whispercpp['wer'].mean(), df_sm4t['wer'].mean()],
    'Time': [df_whisper['time'].mean(), df_whispercpp['time'].mean(), df_sm4t['time'].mean()],
    'Accuracy': [df_whisper['accuracy'].mean(), df_whispercpp['accuracy'].mean(), df_sm4t['accuracy'].mean()]
}

# Converti in DataFrame per una gestione più comoda
means_df = pd.DataFrame(means)

# Funzione per creare un grafico a barre comparativo delle medie
def create_comparative_bar_plot(means_df, metric, title, ylabel, colors):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(means_df['model'], means_df[metric], color=colors, edgecolor='black')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.yscale('log')  # Utilizza una scala logaritmica sull'asse y
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Aggiungi i valori delle barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()  # Aggiusta automaticamente gli elementi del grafico
    plt.show()

# Crea l'istogramma comparativo per il WER medio
create_comparative_bar_plot(means_df, 'WER', 'Average WER Comparison', 'WER', ['skyblue', 'salmon', 'green'])

# Crea l'istogramma comparativo per il tempo medio
create_comparative_bar_plot(means_df, 'Time', 'Average Time Comparison', 'Time (s)', ['skyblue', 'salmon', 'green'])

# Crea l'istogramma comparativo per l'accuratezza media
create_comparative_bar_plot(means_df, 'Accuracy', 'Average Accuracy Comparison', 'Accuracy', ['skyblue', 'salmon', 'green'])
