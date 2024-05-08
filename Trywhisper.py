import whisper
from datasets import load_dataset, Audio
import numpy as np 
import jiwer
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment


# Disattiva i warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_path(path1, path2):
    split = path1.split('/')
    ret = path1.replace(split[-1],path2)
    return ret


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
def get_audio(filename):
    audio=AudioSegment.from_file(filename)
    
    audioLen=len(audio)//(30*1000) #recupero in n di segmenti 
    return audio,audioLen

def get_dataset_from_file(filename):
    ds=[]
    with open(filename, "r") as f:
        for line in f:
            ds.append(line.strip())
        return ds

def use_model(audio):
    model = whisper.load_model("base")
    start=time.time()
    result = model.transcribe(audio)
    end=time.time()-start
    #print(result["text"])
    return result["text"],end

def calculate_WER(transcription,transcriptionFromModel):
    
    #per calcolare il WER va rimossa la punteggiatura
    transcription=transcription.lower().replace(".","").replace(",","")
    transcriptionFromModel=transcriptionFromModel.lower().replace(".","").replace(",","")

    wer=jiwer.wer(transcription,transcriptionFromModel)

    print("WER:", wer )    
    return wer 

def my_plot(wer,t):
    mean_wer = np.mean(wer)
    mean_time = np.mean(t)

    plt.ylabel('WER')
    plt.plot(wer, label='WER ', color='red',marker="o")
    plt.grid(True) 
    plt.legend()
    plt.axhline(y=mean_wer, color='green', linestyle='--', label=f'Valore Medio: {mean_wer:.2f}')  # Linea orizzontale per il valore medio
    plt.show()
    plt.savefig("fig/werfig.png")
    plt.clf()
    
    plt.ylabel('Time')
    plt.plot(t, label='Time' ,marker="o")
    plt.grid(True) 
    plt.legend()    
    plt.axhline(y=mean_time, color='green', linestyle='--', label=f'Valore Medio: {mean_time:.2f}')
    plt.show()
    plt.savefig("fig/timefig.png")
    plt.clf()
     
def accuracy(l_wer):
    acc=[]
    for i in range(len(l_wer)):
        acc.append(1-l_wer[i])
    return acc 



def main2():
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
            
        

def main():
    wer_list = []
    time_list=[]
    

    dataSet=loadDataset()

    for i in range(len(dataSet["test"])-800):
        audio_path=get_path(dataSet["test"][i]['path'],dataSet["test"][i]['audio']['path']) #funzione per ottenere il path dell'audio
        transcription=dataSet["test"][i]["transcription"]
        ipotesi,t=use_model(audio_path)
        time_list.append(t)
        wer_list.append(calculate_WER(transcription,ipotesi))
        print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))

    print(np.mean(wer_list))
    print(np.mean(time_list))
    my_plot(wer_list,time_list)





if __name__=="__main__":
    main()