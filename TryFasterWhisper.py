#python3.10.12
import numpy as np
from datasets import load_dataset, Audio
import time
from CalcolaWER import calculate_WER,accuracyFromWER
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

from faster_whisper import WhisperModel



wer_list=[]
t_list=[]
acc_list=[]


def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)) 
    return ds

def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret



def main():
    
    model_size = "large-v3"
    #compute_type effettua quantizazione
    model = WhisperModel(model_size, device="cpu")

    dataset=loadDataset()
    
    wer_list = []    
    time_list=[]
    acc_list = []

    #INFERENZA
    for i in range(len(dataset["test"])):

        transcription=dataset["test"][i]["transcription"]
        audio_path=get_path(dataset["test"][i]['path'],dataset["test"][i]['audio']['path']) 
        
        t_start=time.time()
        segments, _= model.transcribe(audio_path,language="it")
        t_end=time.time()
        t=t_end-t_start
        
        #recupero trascrizione
        segments = list(segments)        
        ipotesi=segments[0].text
        
        #calcolo WER e accuracy
        wer=calculate_WER(transcription,ipotesi)
        acc=accuracyFromWER(wer)

        wer_list.append(wer)
        time_list.append(t)
        acc_list.append(acc)

        print("Iterazione {}".format(i))



    #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
    WriteMeanToCSV("means.csv","Faste-base",avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
    WriteValues("faster_whisper_base.csv",wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     


    


if __name__ == "__main__":
    main()