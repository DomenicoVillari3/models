from TTS.api import TTS
from datasets import load_dataset
import time

def LoadModel():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    return tts

def load_ds():
    ds = load_dataset("facebook/multilingual_librispeech", "italian",split="train")

    return ds
    

#Trascrizione, prende in input audio (array associato)
def use_model(tts,t):
    # genera trascrizione
    start=time.time()
    # generate speech by cloning a voice using default settings
    tts.tts_to_file(text=t,
                file_path="output.wav",
                speaker_wav="test.wav",
                language="it")
    end=time.time()
    elapsed = end-start

    return elapsed


def main():
    # Carico il modello e il dataset
    model=LoadModel()
    dataset=load_ds()
    


    #INFERENZA
    
       
    

    


if __name__ == "__main__":
    main()