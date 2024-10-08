import os
import csv
import sacrebleu

def WriteMeanToCSV(filename,modello,avg_wer,avg_time,avg_accuracy):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['modello', 'avg_wer','avg_time','avg_accuracy']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            #scrivo riga
            writer.writerow({'modello': modello, 'avg_wer': avg_wer, 'avg_time': avg_time, 'avg_accuracy': avg_accuracy})

def WriteValues(filename,wer_l,time_l,accuracy_l):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['entry' ,'wer','time','accuracy']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            for i in range(len(wer_l)):
                 writer.writerow({'entry': i, 'wer': wer_l[i], 'time': time_l[i], 'accuracy': accuracy_l[i]})

def writeBleu(filename,model,bleu,avg_time):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['model', 'bleu_score','avg_time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            #scrivo riga
            writer.writerow({'model': model, 'bleu_score': bleu, 'avg_time': avg_time})
