#python 3.11.9
import os
from datasets import load_dataset, Audio
import numpy as np 
import time
from pydub import AudioSegment
import subprocess
import soundfile

from CalcolaWER import calculate_WER,accuracyFromWER
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

'''./main --model /home/domenico/whisper.cpp/models/ggml-base.bin
--file samples/jfk.wav --language en --output-txt 
--output-file ./samples/f
'''

#Path to whisper dirercory
target_directory = '/home/domenico/whisper.cpp' #path della directory con il modello   
model_directory = '/home/domenico/whisper.cpp/models/ggml-base.bin' #path del modello 
workspace_directory = '/home/domenico/Scrivania/Models/models' #directory corrente su cui verranno salati i csv
language='it'
output_file='/JobDir/f.txt' #file per salvare la trascrizione 


#Carico dataset
def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=1600)) 
    return ds

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
def use_model(audio):
    global model_directory
    global language
    global output_file
    global target_directory

    to_16bit_wav(audio)

    comando='./main --model {} --file {} --language {} --output-txt --output-file .{} -ng True'.format(model_directory,audio,language,output_file)
    #print(comando)
    
    #ESEGUO IL COMANDO IN UNA SHELL DEDICATA
    try:
        start=time.time()
        output = subprocess.check_output(
            comando, shell=True, stderr=subprocess.STDOUT)
        end=time.time()
        elapsed_time = end - start

    except subprocess.CalledProcessError as e:
        print("Error:", e.output)
        return None, None
    
    #RECUPERO IL CONTENUTO DELLA TRASCRIZIONE DA UN FILE
    file=target_directory+output_file
    with open(file,"r") as f:
        transcription=f.read()
        f.close()
    return transcription,elapsed_time



def main():
    global target_directory
    global workspace_directory

    dataSet=loadDataset() 

    os.chdir(target_directory)
    wer_list = []
    time_list=[]
    acc_list = []
    for i in range(len(dataSet["test"])):
        #print("inizio")
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path'])
        #print("trascrizione")
        transcription=dataSet["test"][i]["transcription"]
        ipotesi,t=use_model(audio_path)
        #print(ipotesi)
        
        time_list.append(t)
        wer=calculate_WER(transcription,ipotesi)
        wer_list.append(wer)
        acc_list.append(accuracyFromWER(wer))
        print("\n ipotesi: {}\ntrascrizione: {} \n, tempo {} \n ".format(ipotesi, transcription,t))
        
        print("Iterazione {}".format(i))

    os.chdir(workspace_directory)
    #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
    WriteMeanToCSV("means.csv",modello="whispercpp_base_cmd_2",avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
    WriteValues("whisperCPP_base_cmd.csv",wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    
    #my_plot("whispercpp.csv","whispercpp")


if __name__=="__main__":
    main()





