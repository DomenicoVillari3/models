import os
from datasets import load_dataset, Audio
import numpy as np 
import time
from pydub import AudioSegment
import subprocess
import soundfile

from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot,plot_accuracy

'''./main --model /home/domenico/whisper.cpp/models/ggml-base.bin
--file samples/jfk.wav --language en --output-txt 
--output-file ./samples/f
'''

#Path to whisper dirercory
target_directory = '/home/domenico/whisper.cpp'
model_directory = '/home/domenico/whisper.cpp/models/ggml-base.bin'
workspace_directory = '/home/domenico/Scrivania/Models/models'
language='it'
output_file='/JobDir/f'



def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=1600)) 
    return ds

def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret

#CONVERSIONE A 16 BIT PER WHISPER CPP
def to_16bit_wav(filepath):
    data, samplerate = soundfile.read(filepath)
    soundfile.write(filepath, data, samplerate, subtype='PCM_16')

def use_model(audio):
    global model_directory
    global language
    global output_file
    global target_directory

    to_16bit_wav(audio)

    comando='./main --model {} --file {} --language {} --output-txt --output-file .{}'.format(model_directory,audio,language,output_file)
    #print(comando)
    
    try:
        start=time.time()
        output = subprocess.check_output(
            comando, shell=True, stderr=subprocess.STDOUT)
        end=time.time()
        elapsed_time = end - start

    except subprocess.CalledProcessError as e:
        print("Error:", e.output)
        return None, None
    
    file=target_directory+output_file+".txt"
    with open(file,"r") as f:
        transcription=f.read()
        f.close()
    return transcription,elapsed_time



def main():
    global target_directory
    global workspace_directory

    wer_list = []
    acc_list = []
    time_list=[]
    

    dataSet=loadDataset()
    os.chdir(target_directory)

    for i in range(len(dataSet["test"])):
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path'])
        transcription=dataSet["test"][i]["transcription"]
        ipotesi,t=use_model(audio_path)
        time_list.append(t)
        wer_list.append(calculate_WER(transcription,ipotesi))
        acc_list.append(accuracyFromWER(wer_list[i]))
        print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))

    os.chdir(workspace_directory)
    my_plot(wer_list,time_list,"whispercpp")
    plot_accuracy("whispercpp",acc_list)


if __name__=="__main__":
    main()





