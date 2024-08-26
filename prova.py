import whisper_cpp_python 
#python 3.11.9
import os
from datasets import load_dataset, Audio
import numpy as np 
import re
import time
from pydub import AudioSegment
import subprocess
import soundfile
import gc

from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

'''./main --model /home/domenico/whisper.cpp/models/ggml-base.bin
--file samples/jfk.wav --language en --output-txt 
--output-file ./samples/f
'''


#Carico dataset
def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=1600)) 
    return ds

def clean(stringa):
    cleaned_string = re.sub(r'[^\w\s]', '', stringa)  # Rimuove le punteggiature e le virgolette
    cleaned_string = cleaned_string.lower()  # Converte tutto in minuscolo
    return cleaned_string


#Recupero del path dell'audio
def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret


#TRASCRIZIONE
def use_model(model,audio):
    start = time.time()
    result = model.transcribe(audio, language="it")
    end = time.time() - start
    #print(result)
    return result["text"], end



def main():
    model = whisper_cpp_python.Whisper(model_path="/home/domenico/whisper.cpp/models/ggml-base.bin",n_threads=4)

    dataSet=loadDataset() 

    wer_list = []
    time_list=[]
    acc_list = []
    for i in range(len(dataSet["test"])):
        #print("inizio")
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path'])
        #print("trascrizione")
        transcription=dataSet["test"][i]["transcription"]
        
        ipotesi,t=use_model(model,audio_path)
        ipotesi = clean(ipotesi)
        #print(ipotesi)
        
        time_list.append(t)
        wer=calculate_WER(transcription,ipotesi)
        wer_list.append(wer)
        acc_list.append(accuracyFromWER(wer))
        print("\n ipotesi: {}\ntrascrizione: {}, \n tempo: {} \n".format(ipotesi, transcription,t))
        
        print("Iterazione {}".format(i))
       


    #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
    WriteMeanToCSV("means.csv",modello="whispercpp_base_api",avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
    WriteValues("whisperCPP_base_api.csv",wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    
    #my_plot("whispercpp.csv","whispercpp")


if __name__=="__main__":
    main()





