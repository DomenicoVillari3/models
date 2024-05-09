import whisper
from datasets import load_dataset, Audio
import numpy as np 
import time
from pydub import AudioSegment

from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot

# Disattiva i warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)) 
    return ds
    
    # load audio sample on the fly
    #audio_input = ds["train"][10]  
    #transcription = ds["train"][10]["transcription"]  # first transcription
    
    #audio_path=get_path(audio_input['path'],audio_input['audio']['path']) #funzione per ottenere il path dell'audio
    
    #print(audio_input)
    #print(transcription)
    #ipotesi=use_model(audio_path)

    #calculate_WER(transcription,ipotesi)

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
    wer_list = []
    time_list=[]
    

    dataSet=loadDataset()

    #PROVA DI TRASCRIZIONI SUL DATASET 
    for i in range(len(dataSet["test"])):
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path']) #funzione per ottenere il path dell'audio
        transcription=dataSet["test"][i]["transcription"]
        ipotesi,t=use_model(audio_path)
        time_list.append(t)
        wer_list.append(calculate_WER(transcription,ipotesi))
        print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))

    #print(np.mean(wer_list))
    #print(np.mean(time_list))
    my_plot(wer_list,time_list)


if __name__=="__main__":
    main()

#####################################################################################################################



#FUNZIONE PER OTTENERE L'AUDIO DAL FILE 
def get_audio(filename):
    audio=AudioSegment.from_file(filename)
    
    audioLen=len(audio)//(30*1000) #recupero in n di segmenti 
    return audio,audioLen

#FUNZIONE PER OTTENERE ! DATASET DA ! FILE TESTUALE
def get_dataset_from_file(filename):
    ds=[]
    with open(filename, "r") as f:
        for line in f:
            ds.append(line.strip())
        return ds
    
def main2():
    wer_list = []
    time_list=[]
    
    interval= 1000*30

    audio,audioLen=get_audio("audio/edipo")

    for i in range(0,(audioLen+1)):
        audio_segment=audio[interval*i:interval*(i+1)]
        audio_segment.export("audio/auidioSegment.wav",format="wav")
        ipotesi,t=use_model("audio/auidioSegment.wav")
        time_list.append(t)