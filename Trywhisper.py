import whisper
from datasets import load_dataset, Audio
import numpy as np 
import time
from pydub import AudioSegment
import csv
import os

from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

# Disattiva i warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)) 
    return ds
    

def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret



#FUNZIONE PER UTILIZZARE WHISPER
def use_model(audio):
    model = whisper.load_model("base")
    start=time.time()
    result = model.transcribe(audio)
    end=time.time()-start
    #print(result["text"])
    return result["text"],end



def main():
    
    dataSet=loadDataset()
    
    for run in range(15):
        wer_list = []
        time_list=[]
        acc_list = []
        #PROVA DI TRASCRIZIONI SUL DATASET 
        for i in range(len(dataSet["test"])//2):
            transcription=dataSet["test"][i]["transcription"]
            #funzione per ottenere il path dell'audio
            audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path']) 
            #Funzione per fare la trascrizione tramite il modello
            ipotesi,t=use_model(audio_path)


            time_list.append(t)
            wer_list.append(calculate_WER(transcription,ipotesi))
            acc_list.append(accuracyFromWER(wer_list[i]))
            #print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))
            print("Iterazione {} sul run {}".format(i,run))

        #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
        WriteMeanToCSV("whisperMEAN.csv",run,avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
        WriteValues("whisper.csv",run,wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    #print(np.mean(wer_list))
    #print(np.mean(time_list))
    my_plot("whisper.csv","whisper")
    



if __name__=="__main__":
    #ds=loadDataset()
    #print(ds)
    main()
