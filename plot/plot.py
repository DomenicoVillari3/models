import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Leggi i file CSV
df_whisper = pd.read_csv("whisper.csv", usecols=["wer", "time"])
df_whisper_cpp = pd.read_csv("whispercpp.csv", usecols=["wer", "time"])
df_sm4t = pd.read_csv("SM4T.csv", usecols=["wer", "time"])

# Prepara i dati per il grafico a barre
labels = ['Time One Iter', 'WER One Iter', 'Time Full', 'WER Full']
whisper_means = [
    np.mean(df_whisper.head(n=433)["time"]),
    np.mean(df_whisper.head(n=433)["wer"]),
    np.mean(df_whisper["time"]),
    np.mean(df_whisper["wer"])
]
whisper_cpp_means = [
    np.mean(df_whisper_cpp.head(n=433)["time"]),
    np.mean(df_whisper_cpp.head(n=433)["wer"]),
    np.mean(df_whisper_cpp["time"]),
    np.mean(df_whisper_cpp["wer"])
]
s_means = [
    np.mean(df_sm4t.head(n=433)["time"]),
    np.mean(df_sm4t.head(n=433)["wer"]),
    np.mean(df_sm4t["time"]),
    np.mean(df_sm4t["wer"])
]

x = np.arange(len(labels))  # Posizione delle etichette
width = 0.25  # Larghezza delle barre

fig, ax = plt.subplots()

bars1 = ax.bar(x - width, whisper_means, width, label='Whisper')
bars2 = ax.bar(x, whisper_cpp_means, width, label='Whisper_cpp')
bars3 = ax.bar(x + width, s_means, width, label='SeamlessM4T')  

# Aggiungi le etichette e il titolo
ax.set_xlabel('Metrica')
ax.set_ylabel('Valore')
ax.set_title('Confronto tra Whisper, Whisper_cpp e SM4T')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Aggiungi la griglia
ax.grid(True)

# Aggiungi le etichette di valore alle barre
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Mostra il grafico
plt.tight_layout()
plt.show()
