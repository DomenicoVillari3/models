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

#CONVERSIONE A 16 BIT PER WHISPER CPP
def to_16bit_wav(filepath):
    data, samplerate = soundfile.read(filepath)
    soundfile.write(filepath, data, samplerate, subtype='PCM_16')

#TRASCRIZIONE
def use_model(model,audio):
    start = time.time()
    result = model.transcribe(audio, language="it",response_format='verbose_json')
    end = time.time() - start
    print(result)
    return result["text"], end



def main():
    global target_directory
    global workspace_directory

    model = whisper_cpp_python.Whisper(model_path="/home/domenico/whisper.cpp/models/ggml-base.bin", n_threads=4)

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
        print("\n ipotesi: {}\ntrascrizione: {}, \n tempo {} \n".format(ipotesi, transcription,t))
        
        print("Iterazione {}".format(i))


    #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
    WriteMeanToCSV("means.csv",modello="whispercpp",avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
    WriteValues("whisperCPP.csv",wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    
    #my_plot("whispercpp.csv","whispercpp")


if __name__=="__main__":
    main()





