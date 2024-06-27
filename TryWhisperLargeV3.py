from datasets import load_dataset, Audio
import numpy as np 
import time
import torch
from transformers import pipeline,AutoModelForSpeechSeq2Seq,AutoProcessor
import csv
import os

from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

# Disattiva i warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def load_model(model_id):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    #use_safetensors=True indica che il modello dovrebbe utilizzare SafeTensors, 
    # che sono un tipo speciale di tensori PyTorch progettati per ridurre l’uso della memoria.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    #modello con chunk di 25 secondi l'uno su batch con 8 elementi (per l'esecuzione in parallelo)
    #se torch_dtype=torch.float16 si ha  dati a virgola mobile a 16 bit, anche detta half precision (maggiore velocità e - memoria)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe

def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)) 
    return ds
    

def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret



#FUNZIONE PER UTILIZZARE WHISPER
def use_model(model,audio):
    start=time.time()
    result = model(audio,generate_kwargs={"language": "italian"})
    end=time.time()-start
    #print(result["text"])
    return result["text"],end



def main():
    modelName="whisper-large-v3"
    modelId="openai/whisper-large-v3"

    dataSet=loadDataset()
    pipe=load_model(modelId)
    
    wer_list = []
    time_list=[]
    acc_list = []
    
    
    #PROVA DI TRASCRIZIONI SUL DATASET 
    for i in range(len(dataSet["test"])):

        #Recupero la trascrizione 
        transcription=dataSet["test"][i]["transcription"]
        print("trascrizione originale {}".format(transcription))

        #funzione per ottenere il path dell'audio
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path']) 
        #print("path dell'audio {}".format(audio_path))


        #Funzione per fare la trascrizione tramite il modello
        ipotesi,t=use_model(pipe,audio_path)
        print("Inferenza del modello {}".format(ipotesi))

        #appendo su liste
        time_list.append(t)
        wer_list.append(calculate_WER(transcription,ipotesi))
        acc_list.append(accuracyFromWER(wer_list[i]))
        
        #print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))
        print("Iterazione {} sul run".format(i))

    #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
    WriteMeanToCSV("means.csv",modelName,avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
    WriteValues(modelName+".csv",wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    #print(np.mean(wer_list))
    #print(np.mean(time_list))
    #my_plot("whisper.csv","whisper")
    



if __name__=="__main__":
    main()
