import os
import csv
def WriteMeanToCSV(filename,esecuzione,avg_wer,avg_time,avg_accuracy):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['esecuzione', 'avg_wer','avg_time','avg_accuracy']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            #scrivo riga
            writer.writerow({'esecuzione': esecuzione, 'avg_wer': avg_wer, 'avg_time': avg_time, 'avg_accuracy': avg_accuracy})

def WriteValues(filename,esecuzione,wer_l,time_l,accuracy_l):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['esecuzione','entry' ,'wer','time','accuracy']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            for i in range(len(wer_l)):
                 writer.writerow({'esecuzione': esecuzione, 'entry': i, 'wer': wer_l[i], 'time': time_l[i], 'accuracy': accuracy_l[i]})